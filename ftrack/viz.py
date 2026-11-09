"""Drawing: scene overlay, upright rotation, and the live dashboard (camera view + strip charts + status)."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

ROT = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]


def upright(img, sc):
    k = sc.rot90_k()
    return img if k == 0 else cv2.rotate(img, ROT[k])


def draw_scene(img, sc, thick=2):
    for e in sc.edges_px:
        cv2.polylines(img, [e.astype(np.int32)], True, (200, 60, 0), thick)
    for c in sc.centre_px:
        cv2.polylines(img, [c.astype(np.int32)], sc.path.closed, (0, 255, 0), thick)
    if sc.board_px is not None:
        cv2.polylines(img, [sc.board_px.astype(np.int32)], True, (90, 90, 90), thick)
    if sc.obstacle_px is not None:
        cv2.polylines(img, [sc.obstacle_px.astype(np.int32)], True, (255, 255, 0), thick + 1)
    for b in sc.blue_px:
        cv2.circle(img, tuple(np.int32(b)), 14, (255, 0, 0), thick)
    # start + direction arrow
    p0 = sc.to_px(sc.path.pts[0])
    p1 = sc.to_px(sc.path.point(25.0))
    cv2.arrowedLine(img, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 255, 255), thick + 2, tipLength=0.4)
    # world axes at the origin (magenta = +Y up, orange = +X right)
    o = sc.to_px([0, 0])
    cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(sc.to_px([0, 30]))), (255, 0, 255), thick + 1)
    cv2.arrowedLine(img, tuple(np.int32(o)), tuple(np.int32(sc.to_px([30, 0]))), (0, 128, 255), thick + 1)
    return img


class Strip:
    """Rolling time-series plot drawn with OpenCV (fast enough for every frame)."""

    def __init__(self, title, labels, colors, span_s=20.0, yrange=None, w=640, h=170, band=None):
        self.title, self.labels, self.colors = title, labels, colors
        self.span, self.yr, self.w, self.h, self.band = span_s, yrange, w, h, band
        self.t = deque()
        self.y = [deque() for _ in labels]

    def add(self, t, *vals):
        self.t.append(t)
        for q, v in zip(self.y, vals):
            q.append(np.nan if v is None else float(v))
        while self.t and t - self.t[0] > self.span:
            self.t.popleft()
            for q in self.y:
                q.popleft()

    def render(self):
        img = np.full((self.h, self.w, 3), 28, np.uint8)
        cv2.putText(img, self.title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if len(self.t) < 2:
            return img
        t = np.array(self.t)
        Y = [np.array(q) for q in self.y]
        if self.yr:
            lo, hi = self.yr
        else:
            allv = np.concatenate([y[np.isfinite(y)] for y in Y]) if any(np.isfinite(y).any() for y in Y) else np.array([0, 1])
            lo, hi = float(np.min(allv)), float(np.max(allv))
            pad = 0.1 * (hi - lo + 1e-6)
            lo, hi = lo - pad, hi + pad
        x0, x1, y0, y1 = 44, self.w - 8, 26, self.h - 18
        X = x0 + (t - (t[-1] - self.span)) / self.span * (x1 - x0)

        def ypix(v):
            return y1 - (v - lo) / (hi - lo + 1e-9) * (y1 - y0)

        if self.band is not None:
            cv2.rectangle(img, (x0, int(ypix(self.band))), (x1, int(ypix(-self.band))), (45, 70, 45), -1)
        if lo < 0 < hi:
            cv2.line(img, (x0, int(ypix(0))), (x1, int(ypix(0))), (90, 90, 90), 1)
        for v in (lo, hi):
            cv2.putText(img, f"{v:.0f}", (2, int(ypix(v)) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        for y, c, lab_, k in zip(Y, self.colors, self.labels, range(len(Y))):
            ok = np.isfinite(y)
            pts = np.column_stack([X, ypix(np.clip(y, lo, hi))]).astype(np.int32)
            seg = []
            for p, g in zip(pts, ok):
                if g:
                    seg.append(p)
                elif len(seg) > 1:
                    cv2.polylines(img, [np.array(seg)], False, c, 1, cv2.LINE_AA)
                    seg = []
                else:
                    seg = []
            if len(seg) > 1:
                cv2.polylines(img, [np.array(seg)], False, c, 1, cv2.LINE_AA)
            last = y[ok][-1] if ok.any() else np.nan
            cv2.putText(img, f"{lab_} {last:+.1f}", (self.w - 150 - 0 * k, 18 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        return img


class Dashboard:
    """Left: upright camera view with path, drone, reference, laser, trail. Right: strip charts + status text."""

    def __init__(self, sc, title="Tello path tracking", view_h=720, record: str | None = None, show=True):
        self.sc, self.title, self.view_h, self.show = sc, title, view_h, show
        self.trail = deque(maxlen=600)
        self.err = Strip("cross-track error e_n (cm)", ["e_n"], [(0, 200, 255)], yrange=(-40, 40), band=5.0)
        self.xy = Strip("position error (cm)  ex / ey", ["ex", "ey"], [(80, 160, 255), (120, 255, 120)], yrange=(-60, 60))
        self.cmd = Strip("rc command  a (L/R)  c (U/D)  b (F/B)", ["a", "c", "b"], [(80, 160, 255), (120, 255, 120), (200, 200, 200)], yrange=(-100, 100))
        self.base = None
        self.writer = None
        self.record = record

    def update_base(self, frame):
        self.base = None

    def render(self, frame, st: dict):
        sc = self.sc
        img = frame.copy()
        draw_scene(img, sc)
        if st.get("uv") is not None and np.all(np.isfinite(st["uv"])):
            self.trail.append(tuple(np.int32(st["uv"])))
        else:
            self.trail.append(None)
        pts = list(self.trail)
        for p, q in zip(pts[:-1], pts[1:]):
            if p and q:
                cv2.line(img, p, q, (0, 0, 255), 3)
        if st.get("box") is not None:
            x0, y0, x1, y1 = np.int32(st["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if st.get("ref_xy") is not None:
            r = sc.to_px(st["ref_xy"])
            cv2.circle(img, tuple(np.int32(r)), 10, (255, 0, 255), 3)
            if st.get("uv") is not None and np.all(np.isfinite(st["uv"])):
                cv2.line(img, tuple(np.int32(st["uv"])), tuple(np.int32(r)), (255, 0, 255), 2)
        if st.get("laser_uv") is not None:
            cv2.drawMarker(img, tuple(np.int32(st["laser_uv"])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
        if st.get("land_zone_xy") is not None:
            cv2.circle(img, tuple(np.int32(sc.to_px(st["land_zone_xy"]))), int(st.get("land_r", 15) / sc.s_px), (0, 165, 255), 3)
        if st.get("uv") is not None and np.all(np.isfinite(st["uv"])):
            cv2.circle(img, tuple(np.int32(st["uv"])), 9, (0, 255, 255), -1)
        view = upright(img, sc)
        s = self.view_h / view.shape[0]
        view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = st.get("t", 0.0)
        self.err.add(t, st.get("e_n"))
        self.xy.add(t, *(st.get("e_xy") if st.get("e_xy") is not None else (None, None)))
        rc = st.get("rc", (0, 0, 0, 0))
        self.cmd.add(t, rc[0], rc[2], rc[1])
        W = 640
        panel = np.full((self.view_h, W, 3), 18, np.uint8)
        y = 0
        for strip in (self.err, self.xy, self.cmd):
            r = strip.render()
            panel[y:y + r.shape[0]] = r
            y += r.shape[0] + 4
        lines = [
            f"{self.title}",
            f"phase: {st.get('phase', '')}    t = {t:6.2f} s",
            f"pos (cm): X {st.get('xy', (np.nan, np.nan))[0]:7.1f}  Y {st.get('xy', (np.nan, np.nan))[1]:7.1f}",
            f"e_n {st.get('e_n', np.nan):+6.1f} cm   RMS(track) {st.get('rms', np.nan):5.1f} cm   max {st.get('emax', np.nan):5.1f}",
            f"progress {st.get('progress', 0) * 100:5.1f} %   v_ref {st.get('v_ref', 0):4.0f} cm/s",
            f"battery {st.get('bat', np.nan):.0f} %   loop {st.get('fps', 0):4.1f} Hz   vision {st.get('lat_ms', 0):4.0f} ms",
            f"{st.get('msg', '')}",
            "keys: SPACE/q land   p pause/resume   ESC emergency stop",
        ]
        for i, s_ in enumerate(lines):
            col = (0, 220, 255) if i == 0 else (235, 235, 235)
            if i == 6:
                col = (80, 80, 255)
            cv2.putText(panel, s_, (10, y + 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        out = np.hstack([view, panel])
        if self.record:
            if self.writer is None:
                self.writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (out.shape[1], out.shape[0]))
            self.writer.write(out)
        key = -1
        if self.show:
            cv2.imshow(self.title, out)
            key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
