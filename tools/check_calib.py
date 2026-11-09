"""Draw the auto-calibration result on a frame (image file, video file at --t seconds, or camera index)."""
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.scene import calibrate
from ftrack.viz import draw_scene, upright

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, choices=["circle", "square", "path"])
ap.add_argument("--src", required=True)
ap.add_argument("--t", type=float, default=0.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
if a.src.isdigit():
    cap = cv2.VideoCapture(int(a.src)); [cap.read() for _ in range(10)]; ok, im = cap.read()
elif a.src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
    im = cv2.imread(a.src)
else:
    cap = cv2.VideoCapture(a.src); cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000); ok, im = cap.read()
t = time.time()
sc = calibrate(im, a.kind)
print(f"{a.kind}: {time.time() - t:.2f}s  up(img)={sc.up.round(3)}  rot90={sc.rot90_k()}  cm/px={sc.s_px:.4f}  length={sc.path.length:.1f} cm")
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.info.items() if k != "homography"})
lo, hi = sc.bounds_cm(0)
print("board (cm):", lo.round(0), hi.round(0), " blue squares (cm):", [sc.to_cm(b).round(0).tolist() for b in sc.blue_px])
vis = upright(draw_scene(im.copy(), sc), sc)
out = a.out or f"calib_{a.kind}.jpg"
cv2.imwrite(out, cv2.resize(vis, None, fx=0.5, fy=0.5))
print("wrote", out)
