"""Offline check of the vision module on a recorded video: drone + laser per frame -> CSV and a contact sheet."""
import argparse, csv, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.scene import calibrate
from ftrack.vision import DroneVision, LaserDetector

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--kind", required=True)
ap.add_argument("--detector", default="bsm"); ap.add_argument("--model_dir", default="models/final_det")
ap.add_argument("--calib_t", type=float, default=0.0); ap.add_argument("--out", default="vision_test")
ap.add_argument("--max_s", type=float, default=1e9); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args()
cap = cv2.VideoCapture(a.video); fps = cap.get(5)
cap.set(cv2.CAP_PROP_POS_MSEC, a.calib_t * 1000); ok, f0 = cap.read(); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
sc = calibrate(f0, a.kind)
vis = DroneVision({"detector": a.detector, "model_dir": a.model_dir}, sc)
lo, hi = sc.bounds_cm(40)
las = LaserDetector(sc.to_px(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])), sc.s_px)
rows, tiles, i = [], [], 0
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
while True:
    ok, f = cap.read()
    if not ok or i / fps > a.max_s: break
    if i % a.stride: i += 1; continue
    t = i / fps
    o = vis(f, t); L = las(f)
    xy = sc.to_cm(o["uv"]) if o["uv"] is not None else (np.nan, np.nan)
    lxy = sc.to_cm(L) if L is not None else (np.nan, np.nan)
    bx = o["box"] if o["box"] is not None else (np.nan,) * 4
    rows.append([i, t, *(o["uv"] if o["uv"] is not None else (np.nan, np.nan)), *xy, *(L if L is not None else (np.nan, np.nan)), *lxy, o["conf"], o["ms"], *bx])
    if (o["uv"] is not None or L is not None) and i % max(1, int(fps * 3)) == 0 and len(tiles) < 24:
        c = (o["uv"] if o["uv"] is not None else L).astype(int)
        crop = f[max(0, c[1] - 120):c[1] + 120, max(0, c[0] - 120):c[0] + 120].copy()
        crop = cv2.copyMakeBorder(crop, 0, 240 - crop.shape[0], 0, 240 - crop.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(crop, f"{t:.1f}s {'D' if o['uv'] is not None else ''}{'L' if L is not None else ''}", (3, 16), 0, 0.5, (0, 255, 255), 1)
        tiles.append(crop)
    i += 1
A = np.array(rows, float)
with open(out / (Path(a.video).stem + f"_{a.detector}.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t", "u", "v", "X", "Y", "lu", "lv", "lX", "lY", "conf", "ms", "bx0", "by0", "bx1", "by1"]); w.writerows(rows)
print(f"{Path(a.video).stem}: {len(A)} frames, drone {100*np.isfinite(A[:,2]).mean():.1f}%, laser {100*np.isfinite(A[:,6]).mean():.1f}%, {np.mean(A[:,11]):.1f} ms/frame")
if tiles:
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / (Path(a.video).stem + f"_{a.detector}_tiles.jpg")), np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]))
