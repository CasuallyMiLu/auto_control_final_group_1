"""Image processing: drone position (YOLO26 + ROI background subtraction + Kalman, from Lab2) and the green laser dot.

DroneVision(cfg)(frame, t) -> {"uv": (u, v) px or None, "box": xyxy or None, "conf", "ms", "coast"}
  detector "yolo": Lab2 real-time tracker (fusion.py), weights from models/<run>/ (engine > onnx > pt)
  detector "bsm" : no-GPU fallback, background subtraction in the work area (dark blob on the white board)
LaserDetector(...)(frame) -> (u, v) px or None
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .fusion import Associator, OnlineBackground


def _work_mask(shape, poly_px, pad_px=0):
    m = np.zeros(shape[:2], np.uint8)
    if poly_px is None:
        m[:] = 1
        return m
    cv2.fillConvexPoly(m, cv2.convexHull(np.int32(poly_px)), 1)
    if pad_px > 0:
        m = cv2.dilate(m, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    return m


class BSMDetector:
    """Dark moving blob vs a median background (frames before take-off), at half resolution.
    'Darker in any colour channel' (not grey level): over the red band the drone is much darker in R even though
    red and the drone have similar grey values. Returns Lab2-style detections in full-resolution pixels."""

    def __init__(self, work_mask, cm_per_px, warmup=25, thr=45, scale=0.5):
        self.f = scale
        self.mask = cv2.resize(work_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        self.warm, self.thr = warmup, thr
        self.stack, self.bg = [], None
        cmpp = cm_per_px / scale
        self.min_a = (4 / cmpp) ** 2
        self.max_a = (45 / cmpp) ** 2

    def __call__(self, frame):
        f = cv2.resize(frame, None, fx=self.f, fy=self.f, interpolation=cv2.INTER_AREA)
        f = cv2.GaussianBlur(f, (3, 3), 0)
        if self.bg is None:
            self.stack.append(f)
            if len(self.stack) >= self.warm:
                self.bg = np.median(np.stack(self.stack), axis=0).astype(np.float32)
                self.stack = []
            return []
        d = cv2.subtract(self.bg.astype(np.uint8), f)  # saturating: only 'darker than background' survives
        dark = cv2.max(cv2.max(d[..., 0], d[..., 1]), d[..., 2])
        m = cv2.threshold(dark, self.thr, 1, cv2.THRESH_BINARY)[1] & self.mask
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        dets = []
        k = 1 / self.f
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if self.min_a <= a <= self.max_a:
                dets.append({"box": np.array([x, y, x + w, y + h], float) * k, "conf": float(min(1.0, a / (self.min_a * 6))),
                             "seg": (np.nan, np.nan), "fuse": (float(cen[i][0]) * k, float(cen[i][1]) * k), "fill": 1.0})
        upd = np.ones(f.shape[:2], np.uint8)
        for d in dets:
            x0, y0, x1, y1 = (d["box"] * self.f).astype(int)
            upd[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = 0
        cv2.accumulateWeighted(f, self.bg, 0.02, mask=upd)
        dets.sort(key=lambda d: -d["conf"])
        return dets[:5]


class DroneVision:
    def __init__(self, cfg: dict, scene=None, root: Path | None = None):
        self.cfg = cfg
        root = root or Path(__file__).resolve().parent.parent
        self.kind = cfg.get("detector", "yolo")
        self.det = None
        if self.kind == "yolo":
            md = root / cfg["model_dir"]
            try:
                from .fusion import Detector
                tc = json.loads((md / "tracker.json").read_text())
                w = next(str(md / f) for f in ("best.engine", "best.pt", "best.onnx") if (md / f).exists())
                if cfg.get("weights"):
                    w = str(md / cfg["weights"])
                self.det = Detector(w, tc["imgsz"], conf=0.05, half=not w.endswith(".onnx"), device=0)
                self.fc = dict(tc["fusion"])
                self.bgm = OnlineBackground(self.fc["bg_scale"], self.fc["bg_warmup"], self.fc["bg_alpha"], self.fc.get("motion_px", 1.0))
                self.tc = tc
                self.weights = w
            except Exception as e:  # missing model / ultralytics / CUDA: fall back so the mission can still run
                print(f"[vision] YOLO unavailable ({type(e).__name__}: {e}); using background subtraction")
                self.kind = "bsm"
        poly = None
        cmpp = 0.2
        if scene is not None:
            lo, hi = scene.bounds_cm(cfg.get("work_margin_cm", 60))
            poly = scene.to_px(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]))
            cmpp = scene.s_px
        self.cmpp = cmpp
        self.work_poly = poly
        if self.kind == "bsm":
            self.bsm = None  # built on the first frame (needs its shape)
            self.tc = {"mode": "fuse", "conf": 0.2, "kalman": True, "q": 2e7}
            self.fc = {"conf_reacquire": 0.5, "gate_px": 150, "coast": 0, "kalman_r": 9.0}
        tc, fc = self.tc, self.fc
        self.asc = Associator(mode=tc["mode"], conf=cfg.get("conf", tc["conf"]), conf_reacquire=max(tc["conf"], fc["conf_reacquire"]),
                              gate_px=fc["gate_px"], coast=cfg.get("coast", 3), kalman=tc["kalman"], q=tc["q"] or 2e6,
                              r=fc["kalman_r"], static_s=0.0)
        self.min_box_px = cfg.get("min_box_cm", 6) / cmpp
        self.max_box_px = cfg.get("max_box_cm", 60) / cmpp

    def reset_background(self):
        if self.kind == "yolo":
            self.bgm = OnlineBackground(self.fc["bg_scale"], self.fc["bg_warmup"], self.fc["bg_alpha"], self.fc.get("motion_px", 1.0))
        else:
            self.bsm = None

    def __call__(self, frame, t):
        t0 = time.perf_counter()
        if self.kind == "yolo":
            from .fusion import measure
            dets, _ = measure(frame, self.det, self.bgm, self.fc)
        else:
            if self.bsm is None:
                self.bsm = BSMDetector(_work_mask(frame.shape, self.work_poly), self.cmpp)
            dets = self.bsm(frame)
        # physical size gate: a Tello is ~20 cm; reject tiny specks and huge boxes (people walking by)
        keep = []
        for d in dets:
            w, h = d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]
            if self.min_box_px <= max(w, h) <= self.max_box_px:
                keep.append(d)
        o = self.asc.step(keep, t)
        ms = (time.perf_counter() - t0) * 1e3
        uv = None if not np.isfinite(o["x"]) else np.array([o["x"], o["y"]])
        vel = None
        if uv is not None and self.asc.kf.x is not None:
            vel = np.array(self.asc.kf.x[2:4], float)
        d = o["det"]
        return {"uv": uv, "vel_px": vel, "box": None if d is None else d["box"], "conf": 0.0 if d is None else d["conf"],
                "coast": o["coast"], "ms": ms, "ndet": len(dets)}


class LaserDetector:
    """Green laser pointer dot: very bright, green-dominant, small. Searched inside the work area only (the dot
    also flares on the side wall). The centre of a strong dot saturates to white, so a white core next to a green
    halo also counts."""

    def __init__(self, work_poly_px=None, cm_per_px=0.2, min_green=40, min_v=170, max_cm=15.0, scale=0.5):
        self.poly = work_poly_px
        self.mask = None
        self.f = scale
        self.min_green, self.min_v = min_green, min_v
        self.max_area = (max_cm / cm_per_px * scale) ** 2
        self.last = None

    def __call__(self, frame):
        frame = cv2.resize(frame, None, fx=self.f, fy=self.f, interpolation=cv2.INTER_AREA)
        if self.mask is None:
            self.mask = _work_mask(frame.shape, None if self.poly is None else np.asarray(self.poly) * self.f)
        f = frame.astype(np.int16)
        b, g, r = f[..., 0], f[..., 1], f[..., 2]
        green = (g - np.maximum(r, b) > self.min_green) & (g > self.min_v)
        core = (g > 245) & (r > 200) & (b > 200)
        halo = cv2.dilate(green.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        m = ((green | (core & halo)) & (self.mask > 0)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        best, score = None, 0.0
        for i in range(1, n):
            a = st[i, 4]
            if a < 2 or a > self.max_area:
                continue
            sel = lab == i
            sc_ = a * float(np.mean(g[sel] - np.maximum(r[sel], b[sel]) + 0.3 * g[sel]))
            if self.last is not None:  # prefer continuity
                sc_ /= 1 + np.hypot(*(cen[i] / self.f - self.last)) / 400
            if sc_ > score:
                best, score = cen[i].copy() / self.f, sc_
        self.last = best if best is not None else self.last
        return best
