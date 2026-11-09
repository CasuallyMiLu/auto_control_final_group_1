"""List cameras 0..4: resolution, real fps, and whether a board (circle / square / path) is visible.
Saves camera_<i>.jpg next to this script's parent folder. python tools/check_camera.py"""
import sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.scene import calibrate
out = Path(__file__).resolve().parent.parent / "runs"
out.mkdir(exist_ok=True)
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5): cap.read()
    t = time.time(); n = 0
    while time.time() - t < 2.0:
        ok, f = cap.read(); n += ok
    cap.release()
    if not n:
        continue
    found = []
    for k in ("circle", "square", "path"):
        try:
            sc = calibrate(f, k); found.append(f"{k} ({sc.s_px:.3f} cm/px)")
        except Exception:
            pass
    cv2.imwrite(str(out / f"camera_{i}.jpg"), f)
    print(f"camera {i}: {f.shape[1]}x{f.shape[0]}, {n / 2:.1f} fps, board: {', '.join(found) or 'none'}  -> runs/camera_{i}.jpg")
