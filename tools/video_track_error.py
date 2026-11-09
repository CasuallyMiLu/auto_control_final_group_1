"""Offline: drone trajectory of a recorded video (from tools/test_vision_video.py CSV) vs the auto-detected path.
python tools/video_track_error.py --video <mp4> --kind circle --csv <..._yolo.csv> --out fig.png"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.scene import calibrate
ap = argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--kind"); ap.add_argument("--csv"); ap.add_argument("--out"); ap.add_argument("--t", type=float, default=0)
a = ap.parse_args()
cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000); ok, f = cap.read()
sc = calibrate(f, a.kind)
d = pd.read_csv(a.csv)
P = d[["u", "v"]].to_numpy(float); ok = np.isfinite(P).all(1)
XY = sc.to_cm(P[ok])
dist = sc.path.distance(XY)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
C = np.vstack([sc.path.pts, sc.path.pts[:1]]) if sc.path.closed else sc.path.pts
ax[0].plot(C[:, 0], C[:, 1], color="0.4", lw=8, alpha=0.3, label="auto-detected centre line")
ax[0].plot(XY[:, 0], XY[:, 1], ".", ms=1.5, color="tab:red", label="drone (YOLO26 fine-tuned)")
ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_xlabel("X (cm)"); ax[0].set_ylabel("Y (cm)"); ax[0].set_title(Path(a.video).name)
near = dist < 40
ax[1].hist(dist[near], bins=40, color="tab:blue"); ax[1].set_xlabel("distance to centre line (cm), samples within 40 cm")
ax[1].set_title(f"median {np.median(dist[near]):.1f} cm, P90 {np.percentile(dist[near], 90):.1f} cm (n={near.sum()})")
fig.tight_layout(); fig.savefig(a.out, dpi=120)
print(f"{Path(a.video).name}: {ok.mean()*100:.0f}% frames tracked, near path {near.mean()*100:.0f}%, median dist {np.median(dist[near]):.1f} cm")
