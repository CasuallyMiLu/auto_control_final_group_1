"""Cut an RGBA drone sprite out of a recorded flight (difference to a clean frame) for the simulator."""
import sys, json, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.scene import calibrate
video, t_drone, t_clean, kind = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
u0, v0 = int(float(sys.argv[5])), int(float(sys.argv[6]))  # approximate drone pixel position
cap = cv2.VideoCapture(video)
def at(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); return cap.read()[1]
fd = at(t_drone)
fc = np.median(np.stack([at(t_drone + k) for k in (-20, -15, -10, -6, 6, 10, 15, 20)]), axis=0).astype(np.uint8)  # drone moves, background stays
fc0 = at(t_clean)
sc = calibrate(fc0, kind)
d = np.abs(fd.astype(np.int16) - fc.astype(np.int16)).max(2)
m = (d > 40).astype(np.uint8)
win = np.zeros_like(m); win[max(0, v0 - 200):v0 + 200, max(0, u0 - 200):u0 + 200] = 1; m &= win
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(m)
i = 1 + int(np.argmax(st[1:, 4]))
x, y, w, h, _ = st[i]
p = 6
crop = fd[y - p:y + h + p, x - p:x + w + p]
alpha = (lab[y - p:y + h + p, x - p:x + w + p] == i).astype(np.uint8)
alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8)) * 255
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
out = Path(__file__).resolve().parent.parent / "assets"
cv2.imwrite(str(out / "drone_sprite.png"), np.dstack([crop, alpha]))
cv2.imwrite(str(out / f"background_{kind}.jpg"), fc0, [cv2.IMWRITE_JPEG_QUALITY, 92])
json.dump({"cm_per_px": sc.s_px, "size_px": [int(w), int(h)], "source": Path(video).name, "t": t_drone}, open(out / "drone_sprite.json", "w"), indent=1)
print("sprite", w, h, "px  ->", w * sc.s_px, h * sc.s_px, "cm")
