"""Fine-tune dataset: BSM-labelled frames of the final-project videos + laser-only negatives + Lab2 dataset."""
import csv, os, random
from pathlib import Path
import cv2, numpy as np
R = Path.home() / "ml"
OUT = R / "work_final" / "ds_final"
VID = {"19_49_31": ("F_square1", "train"), "19_55_46": ("F_circle1", "train"), "19_57_36": ("F_circle2", "val"),
       "20_08_23": ("F_path1", "val"), "20_10_01": ("F_path2", "train")}
STRIDE = 3
random.seed(0)
for sp in ("train", "val"):
    (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
stats = {}
for k, (name, sp) in VID.items():
    rows = list(csv.DictReader(open(R / "final_bsm" / f"WIN_20261109_{k}_Pro_bsm.csv")))
    box = np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in ("bx0", "by0", "bx1", "by1")] for r in rows])
    uv = np.array([[float(r["u"]) if r["u"] not in ("", "nan") else np.nan, float(r["v"]) if r["v"] not in ("", "nan") else np.nan] for r in rows])
    ok = np.all(np.isfinite(box), 1)
    want = {}
    for i in range(2, len(rows) - 2, STRIDE):
        if not ok[i - 2:i + 3].all():
            continue
        mv = np.hypot(*np.diff(uv[i - 2:i + 3], axis=0).T).max()
        w, h = box[i, 2] - box[i, 0], box[i, 3] - box[i, 1]
        if mv < 30 and 12 < min(w, h) and max(w, h) < 300:
            want[i] = box[i]
    cap = cv2.VideoCapture(str(R / "videos_final" / f"WIN_20261109_{k}_Pro.mp4"))
    i, n = 0, 0
    while want:
        okf, f = cap.read()
        if not okf:
            break
        if i in want:
            H, W = f.shape[:2]
            x0, y0, x1, y1 = want.pop(i)
            pw, ph = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
            x0, y0, x1, y1 = max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)
            stem = f"{name}_{i:06d}"
            cv2.imwrite(str(OUT / "images" / sp / f"{stem}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / sp / f"{stem}.txt").write_text(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
            n += 1
        i += 1
    stats[name] = (sp, n)
# negatives: laser-only video (green dot, people, wall hook), every 2nd frame -> train, empty label files
cap = cv2.VideoCapture(str(R / "videos_final" / "WIN_20261109_20_13_14_Pro.mp4"))
i, n = 0, 0
while True:
    okf, f = cap.read()
    if not okf:
        break
    if i % 2 == 0:
        stem = f"F_laser_{i:06d}"
        cv2.imwrite(str(OUT / "images" / "train" / f"{stem}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / "train" / f"{stem}.txt").write_text("")
        n += 1
    i += 1
stats["F_laser(neg)"] = ("train", n)
lab2 = R / "work" / "dataset" / "det"
(OUT / "data.yaml").write_text(
    f"path: {OUT}\ntrain:\n  - {OUT}/images/train\n  - {lab2}/images/train\nval:\n  - {OUT}/images/val\n  - {lab2}/images/val\nnames:\n  0: drone\n")
print(stats)
