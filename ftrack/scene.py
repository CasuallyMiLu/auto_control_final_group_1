"""Automatic scene calibration from one camera frame.

Finds the red track, its white centre line (circle / square) or its medial axis (theme-2 path), the board, the blue
squares and the black obstacle, and builds the map image px <-> wall cm (X right, Y up, gravity aligned).

Orientation ("auto-upright"): the camera may be mounted at any multiple of 90 deg (plus a small tilt).
  fine angle : dominant edge direction of the track / centre cross (mod 90 deg)
  coarse     : path scene  -> B end (next to a blue square) is the top of the track
               circle/square -> the side of the image with the wood floor is down (colour + texture); the mission
               then confirms it with a short climb after take-off (Scene.set_up_vector) before any lateral flight.
Scale: circle/square from the white line (150 cm), path from the red line width (20 cm). Digital zoom is therefore
irrelevant: everything is re-measured each time a program starts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .path import Path, smooth_polyline

DIRS = {"up": (0.0, -1.0), "right": (1.0, 0.0), "down": (0.0, 1.0), "left": (-1.0, 0.0)}


@dataclass
class Geometry:
    centre_line_cm: float = 150.0   # white centre line: circle diameter / square side
    band_cm: float = 20.0           # total red band width (both red stripes + white line)
    path_width_cm: float = 20.0     # theme-2 red line width
    blue_cm: float = 20.0           # blue square side


@dataclass
class Scene:
    kind: str
    shape: tuple
    origin_px: np.ndarray            # px of world (0, 0)
    up: np.ndarray                   # unit image vector of world +Y
    s_px: float                      # cm per px along each world axis (similarity part)
    A: np.ndarray                    # 2x2 cm <- px (after subtracting origin); general affine for circle/square
    path: Path
    centre_px: list = field(default_factory=list)   # centre line in px (for drawing)
    edges_px: list = field(default_factory=list)    # inner/outer track edges in px (for drawing)
    board_px: np.ndarray | None = None
    blue_px: list = field(default_factory=list)
    obstacle_px: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- mapping
    def to_cm(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float)
        return (uv - self.origin_px) @ self.A.T

    def to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        return xy @ np.linalg.inv(self.A).T + self.origin_px

    def rot90_k(self) -> int:
        """np.rot90/cv2.rotate count that makes the displayed frame upright (world up -> screen up)."""
        ang = math.degrees(math.atan2(self.up[1], self.up[0]))  # up=(0,-1) -> -90
        return int(round((ang + 90) / 90)) % 4

    def set_up_vector(self, up_img: np.ndarray):
        """Re-orient the world frame so that +Y points along up_img (e.g. measured during a climb). Keeps the
        scale/shape fit; only the 90-degree choice is changed (snapped to the nearest of the 4 candidates)."""
        up_img = np.asarray(up_img, float) / (np.linalg.norm(up_img) + 1e-12)
        cands = [_rot(self.up, k * 90) for k in range(4)]
        k = int(np.argmax([np.dot(c, up_img) for c in cands]))
        if k == 0:
            return False
        R = _rot2(-k * 90)  # world coordinates rotate opposite to the image up vector
        self.up = cands[k]
        self.A = R @ self.A
        # the shape is symmetric, so the reference path itself is unchanged except for its start point/direction,
        # which the mission re-derives from config; rebuild the path so its points are in the new frame
        self.path = Path(self.path.pts @ R.T, self.path.closed)
        self.info["up_changed_by_deg"] = k * 90
        return True

    def bounds_cm(self, margin: float = 30.0):
        """Working area (wall cm): track + blue squares, grown by margin. The black obstacle is NOT used in flight
        (theme 2: the drone only follows the laser; the operator steers it around the obstacle)."""
        P = [self.path.pts]
        if self.blue_px:
            P.append(self.to_cm(np.array(self.blue_px)))
        P = np.vstack(P)
        return P.min(0) - margin, P.max(0) + margin


# =============================================================================== helpers
def _rot(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _rot2(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (((h < 10) | (h > 165)) & (s > 90) & (v > 60)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def blue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] > 92) & (hsv[..., 0] < 128) & (hsv[..., 1] > 80) & (hsv[..., 2] > 60)).astype(np.uint8)


def components(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = [(i, st[i]) for i in range(1, n) if st[i, 4] >= min_area]
    out.sort(key=lambda x: -x[1][4])
    return lab, out


def dominant_angle_mod90(mask) -> float:
    """Dominant edge orientation of a binary mask in degrees, folded into [-45, 45)."""
    m = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.2)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sel = mag > 0.2 * mag.max()
    th = np.arctan2(gy[sel], gx[sel])
    w = mag[sel]
    z = np.sum(w * np.exp(4j * th))
    return math.degrees(np.angle(z) / 4)


def board_polygon(bgr, seed_mask=None):
    """The white board: bright, unsaturated, large; return its minAreaRect corners (px) or None."""
    # the board is a cool (slightly bluish) white, the wall a warm beige: LAB b* separates them
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (9, 9), 0), cv2.COLOR_BGR2LAB)
    white = ((lab[..., 0] > 110) & (lab[..., 2] < 126) & (np.abs(lab[..., 1].astype(int) - 128) < 8)).astype(np.uint8)
    if seed_mask is not None:  # the track is painted on the board: count it as board
        white |= seed_mask
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    lab, comps = components(white, 0.05 * white.size)
    if not comps:
        return None
    if seed_mask is not None:  # prefer the component that contains the track
        best = max(comps, key=lambda c: np.count_nonzero(seed_mask[lab == c[0]]))
    else:
        best = comps[0]
    cs, _ = cv2.findContours((lab == best[0]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def floor_side(bgr) -> tuple[str, float]:
    """Which image border is the floor: wood is more saturated and more textured than the wall / ceiling.
    Returns (side, confidence 0..1). 'down' means the floor is at the bottom of the image (camera upright)."""
    small = cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    h, w = g.shape
    b = int(0.07 * min(h, w))
    strips = {"down": (slice(h - b, h), slice(None)), "up": (slice(0, b), slice(None)),
              "left": (slice(None), slice(0, b)), "right": (slice(None), slice(w - b, w))}
    score = {}
    for k, (ys, xs) in strips.items():
        hh, ss, vv = hsv[ys, xs, 0], hsv[ys, xs, 1], hsv[ys, xs, 2]
        woody = ((hh > 5) & (hh < 28) & (ss > 45) & (vv > 60) & (vv < 235)).mean()
        score[k] = woody + 0.01 * tex[ys, xs].mean()
    order = sorted(score, key=score.get, reverse=True)
    conf = (score[order[0]] - score[order[1]]) / (score[order[0]] + 1e-6)
    return order[0], float(np.clip(conf, 0, 1))


OPP = {"down": "up", "up": "down", "left": "right", "right": "left"}


def up_from_hint(hint: str, fine_deg: float) -> np.ndarray:
    return _rot(np.array(DIRS[hint]), fine_deg)


def _affine_from(origin, up, scale_x, scale_y=None):
    """2x2 cm<-px for a similarity (or axis-scaled) map with world +Y along `up` (image vector)."""
    scale_y = scale_x if scale_y is None else scale_y
    right = np.array([-up[1], up[0]])
    return np.array([right * scale_x, up * scale_y])


# =============================================================================== per-scene fits
def calibrate(bgr, kind: str, geom: Geometry | None = None, up_hint: str = "auto", direction: str = "ccw",
              start_angle_deg: float | None = None) -> Scene:
    geom = geom or Geometry()
    if kind == "circle":
        sc = _calib_circle(bgr, geom, up_hint)
    elif kind == "square":
        sc = _calib_square(bgr, geom, up_hint)
    elif kind == "path":
        sc = _calib_path(bgr, geom, up_hint)
    else:
        raise ValueError(kind)
    if kind in ("circle", "square"):
        orient_closed_path(sc, direction, start_angle_deg)
    blues = blue_squares(bgr, sc)
    sc.blue_px = blues
    return sc


def orient_closed_path(sc: Scene, direction: str = "ccw", start_angle_deg: float | None = None):
    """Closed path: travel direction (ccw/cw as seen from the camera) and start vertex (angle from centre, deg,
    0 = right, -90 = bottom). Default start: bottom-right (-45 deg), as on the course sheet."""
    P = sc.path.pts
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if (area > 0) != (direction == "ccw"):
        P = P[::-1]
    a0 = math.radians(-45 if start_angle_deg is None else start_angle_deg)
    c = 0.5 * (P.min(0) + P.max(0))
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    i0 = int(np.argmin(np.abs((ang - a0 + np.pi) % (2 * np.pi) - np.pi)))  # vertex in that direction from the centre
    sc.path = Path(np.roll(P, -i0, axis=0), closed=True)
    sc.info.update(direction=direction, start_angle_deg=math.degrees(a0))


def _track_contours(mask):
    lab, comps = components(mask, 0.002 * mask.size)
    if not comps:
        raise RuntimeError("no red track found: check camera / lighting")
    big = np.isin(lab, [c[0] for c in comps[:2]]).astype(np.uint8)
    cs, _ = cv2.findContours(big, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)
    if len(cs) < 4:
        raise RuntimeError(f"red track: expected 4 edges (2 stripes), found {len(cs)}")
    return cs[:4], big


def _resolve_up(bgr, up_hint, fine):
    if up_hint != "auto":
        return up_from_hint(up_hint, fine), {"up_source": "config", "up_hint": up_hint}
    side, conf = floor_side(bgr)
    return up_from_hint(OPP[side], fine), {"up_source": "floor", "floor_side": side, "floor_conf": conf}


def _calib_circle(bgr, g: Geometry, up_hint):
    m = red_mask(bgr)
    cs, big = _track_contours(m)
    E = [cv2.fitEllipse(c) for c in cs]  # outer, white-outer, white-inner, inner (by area)
    fill = cv2.contourArea(cs[0]) / (math.pi / 4 * E[0][1][0] * E[0][1][1])
    if not 0.95 < fill < 1.05:
        raise RuntimeError(f"the red track is not a circle (area / ellipse = {fill:.2f}); is this the square board?")
    # one ellipse through both edges of the white line = the white centre line
    (cx, cy), (a, b), ang = cv2.fitEllipse(np.vstack([cs[1], cs[2]]))
    centre = np.array([cx, cy])
    # fine rotation from the black centre cross
    r = int(0.25 * min(a, b))
    x0, y0 = int(centre[0] - r), int(centre[1] - r)
    roi = bgr[max(0, y0):y0 + 2 * r, max(0, x0):x0 + 2 * r]
    dark = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) < 90).astype(np.uint8)
    fine = dominant_angle_mod90(dark) if dark.sum() > 50 else 0.0
    up, info = _resolve_up(bgr, up_hint, fine)
    # ellipse -> circle: scale along the ellipse axes, then rotate into the world frame
    th = math.radians(ang)
    U = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])  # px -> ellipse axes
    S = np.diag([g.centre_line_cm / a, g.centre_line_cm / b])
    Mpx = U.T @ S @ U  # px -> "round" px-aligned cm
    A = _affine_from(None, up, 1.0) @ Mpx
    R = g.centre_line_cm / 2
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    path = Path(np.column_stack([R * np.cos(t), R * np.sin(t)]), closed=True)
    sc = Scene("circle", bgr.shape[:2], centre, up, float(np.sqrt(abs(np.linalg.det(A)))), A, path)
    sc.edges_px = [c[:, 0, :].astype(np.float32) for c in (cs[0], cs[3])]
    sc.centre_px = [sc.to_px(path.pts)]
    sc.info.update(info, ellipse_axes_px=[a, b], fine_deg=fine, band_px=(E[0][1][0] - E[3][1][0]) / 2,
                   band_cm_measured=float((E[0][1][0] - E[3][1][0]) / 2 * g.centre_line_cm / a))
    return sc


def _quad(c):
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        q = cv2.approxPolyDP(c, eps * peri, True)
        if len(q) == 4:
            return q[:, 0, :].astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def _calib_square(bgr, g: Geometry, up_hint):
    m = red_mask(bgr)
    cs, big = _track_contours(m)
    (_, (rw, rh), _) = cv2.minAreaRect(cs[0])
    fill = cv2.contourArea(cs[0]) / (rw * rh)
    if not fill > 0.93:
        raise RuntimeError(f"the red track is not a square (area / rectangle = {fill:.2f}); is this the circle board?")
    q1, q2 = _quad(cs[1]), _quad(cs[2])
    centre = np.vstack([q1, q2]).mean(0)

    def order(q):  # sort corners by angle around the centre
        a = np.arctan2(q[:, 1] - centre[1], q[:, 0] - centre[0])
        return q[np.argsort(a)]

    q1, q2 = order(q1), order(q2)
    # pair corners of the two white-line edges by proximity
    q2 = np.array([q2[np.argmin(np.linalg.norm(q2 - p, axis=1))] for p in q1])
    Q = (q1 + q2) / 2  # white line corners (px)
    fine = dominant_angle_mod90(m)
    up, info = _resolve_up(bgr, up_hint, fine)
    right = np.array([-up[1], up[0]])
    h = g.centre_line_cm / 2
    W = np.array([[np.sign(np.dot(p - centre, right)) * h, np.sign(np.dot(p - centre, up)) * h] for p in Q])
    Hm, _ = cv2.findHomography(Q, W)
    # keep an affine (fit by least squares on the 4 corners) so to_cm/to_px stay linear and invertible
    X = np.hstack([Q - centre, np.ones((4, 1))])
    sol, *_ = np.linalg.lstsq(X, W, rcond=None)
    A = sol[:2].T
    origin = centre - np.linalg.solve(A, sol[2])
    path = Path(np.array([[h, -h], [h, h], [-h, h], [-h, -h]]), closed=True)
    sc = Scene("square", bgr.shape[:2], origin, up, float(np.sqrt(abs(np.linalg.det(A)))), A, path)
    sc.edges_px = [cs[0][:, 0, :].astype(np.float32), cs[3][:, 0, :].astype(np.float32)]
    sc.centre_px = [sc.to_px(np.vstack([path.pts, path.pts[:1]]))]
    res = np.linalg.norm((Q - origin) @ A.T - W, axis=1)
    sc.info.update(info, fine_deg=fine, corner_fit_rms_cm=float(np.sqrt(np.mean(res ** 2))), homography=Hm.tolist())
    return sc


def _calib_path(bgr, g: Geometry, up_hint):
    from skimage.morphology import skeletonize
    m = red_mask(bgr)
    lab, comps = components(m, 0.002 * m.size)
    if not comps:
        raise RuntimeError("no red path found")
    track = (lab == comps[0][0]).astype(np.uint8)
    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    f = 0.5  # skeletonize at half resolution (8 s -> <1 s), coordinates scaled back
    small = cv2.resize(track, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform(small, cv2.DIST_L2, 5)
    sk = skeletonize(small > 0)
    ys, xs = np.nonzero(sk)
    dv = np.sort(dist[ys, xs])
    dv = dv[int(0.25 * len(dv)):int(0.75 * len(dv))]  # trimmed mean: ends / corners / junctions excluded
    width_px = 2 * float(dv.mean()) / f + 1.0  # +1: the distance transform measures to the last inside pixel
    s_px = g.path_width_cm / width_px  # cm per px
    line = _longest_skeleton_path(sk) / f  # ordered px points from one end to the other
    line = resample_px(line, 2.0)
    # thinning forks towards the two corners of a square line end: cut one line width off each end, then
    # extend straight along the remaining end direction to the painted end
    n_cut = max(2, int(round(width_px / 2.0)))
    line = line[n_cut:-n_cut]
    line = smooth_polyline(line, max(3, int(0.15 * width_px)), closed=False)
    m_dir = max(2, int(round(0.5 * width_px / 2.0)))
    for end in (0, -1):
        nb = line[m_dir] if end == 0 else line[-1 - m_dir]
        d = line[end] - nb
        d /= np.linalg.norm(d) + 1e-9
        ext = line[end] + d[None] * np.linspace(2.0, width_px * 1.0, max(2, int(width_px / 2)))[:, None]
        line = np.vstack([ext[::-1], line]) if end == 0 else np.vstack([line, ext])
    fine = dominant_angle_mod90(track)
    blues = _blue_blobs(bgr)
    if len(blues) < 2:
        raise RuntimeError(f"theme-2 board needs 2 blue squares, found {len(blues)}; is this the circle/square board?")
    # B = the end next to a blue square; the path runs A -> B; up = roughly A -> B
    info = {}
    if blues:
        dB = [min(np.linalg.norm(line[e] - b["c"]) for b in blues) for e in (0, -1)]
        if dB[0] < dB[1]:
            line = line[::-1]
        info["B_blue_dist_px"] = float(min(dB))
    if up_hint != "auto":
        up = up_from_hint(up_hint, fine)
        info["up_source"] = "config"
    else:
        ab = line[-1] - line[0]
        cands = [up_from_hint(k, fine) for k in DIRS]
        up = cands[int(np.argmax([np.dot(c, ab) for c in cands]))]
        info["up_source"] = "path A->B"
    A = _affine_from(None, up, s_px)
    origin = line[0].copy()  # world origin at A
    sc = Scene("path", bgr.shape[:2], origin, up, s_px, A, None)
    pts_cm = sc.to_cm(line)
    sc.path = Path(pts_cm, closed=False)
    sc.centre_px = [line]
    cs, _ = cv2.findContours(track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sc.edges_px = [max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)]
    # black obstacle: dark, large, compact, inside the hull of track + blue squares
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 50).astype(np.uint8) & (1 - cv2.dilate(track, np.ones((9, 9), np.uint8)))
    hp = np.vstack([np.argwhere(track)[:, ::-1]] + [np.array([b["c"]]) for b in blues]).astype(np.float32)
    lo, hi = hp.min(0), hp.max(0)
    pad = 0.08 * (hi - lo)
    bm = np.zeros_like(dark)
    bm[int(max(0, lo[1] - pad[1])):int(hi[1] + pad[1]), int(max(0, lo[0] - pad[0])):int(hi[0] + pad[0])] = 1
    dark &= bm
    _, dcomps = components(dark, int((10 / s_px) ** 2))
    if dcomps:
        x, y, w, h, _ = dcomps[0][1]
        sc.obstacle_px = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    sc.info.update(info, width_px=width_px, fine_deg=fine, length_cm=sc.path.length)
    return sc


def resample_px(line, step):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    si = np.arange(0, d[-1], step)
    return np.column_stack([np.interp(si, d, line[:, 0]), np.interp(si, d, line[:, 1])])


def _longest_skeleton_path(sk) -> np.ndarray:
    """Pixels of the longest geodesic path in a skeleton (drops the short spurs thinning leaves at corners)."""
    ys, xs = np.nonzero(sk)
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    nbrs = [[] for _ in range(len(ys))]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = idx.get((y + dy, x + dx))
                if j is not None and j != i:
                    nbrs[i].append(j)

    def bfs(src):
        dist = np.full(len(ys), -1)
        prev = np.full(len(ys), -1)
        dist[src] = 0
        q = [src]
        for u in q:
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
        return dist, prev

    d0, _ = bfs(0)
    a = int(np.argmax(d0))
    d1, prev = bfs(a)
    b = int(np.argmax(d1))
    out = [b]
    while out[-1] != a:
        out.append(int(prev[out[-1]]))
    return np.array([(xs[i], ys[i]) for i in out[::-1]], float)


def _blue_blobs(bgr):
    m = cv2.morphologyEx(blue_mask(bgr), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lab, comps = components(m, 0.0004 * m.size)
    out = []
    for i, st in comps[:4]:
        x, y, w, h, area = st
        if 0.5 < w / max(h, 1) < 2.0 and area > 0.6 * w * h:
            out.append({"c": np.array([x + w / 2, y + h / 2], float), "box": (x, y, w, h)})
    return out


def blue_squares(bgr, sc: Scene):
    return [b["c"] for b in _blue_blobs(bgr)]
