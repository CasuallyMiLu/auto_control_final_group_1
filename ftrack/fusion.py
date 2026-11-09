"""Real-time drone tracker: YOLO detector + ROI background subtraction + causal Kalman.

Shared by the offline evaluation (workstation) and the live inference (laptop / MATLAB).
measure()   : detector + per-detection ROI refinement + online background update  (needs pixels)
Associator  : detection selection + Kalman filtering                               (pure python, replayable)

Latency notes (laptop, RTX 4050). Every shortcut below gives the same detections and background as the plain
version, bit for bit, on the TA and own videos:
- The frame is resized once to the detector's letterbox size with cv2 (the exact resize ultralytics' LetterBox would
  do), so the detector only pads. At bg_scale 0.5 with a 960 px detector that resize is also the background frame
  (OpenCV runs an exact 2x INTER_LINEAR as INTER_AREA), so the full frame is resized only once per step.
- TensorRT detection engines bypass ultralytics' predictor: the resized frame is uploaded, padded and normalised on
  the GPU into a persistent input buffer, run through ultralytics' CUDA-graph TensorRT backend, and filtered and
  rescaled with ultralytics' own non_max_suppression and scale_boxes (no Results objects, no per-call cfg parsing).
- The background update of frame i runs in a worker thread while the detector runs on frame i+1, because only the
  ROI refinement of frame i+1 needs it. The warm-up median runs on the GPU instead of np.nanmedian (6 s -> <0.1 s).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np


def letterbox_unpad(shape, imgsz: int):
    """(w, h) of the resized, unpadded image inside ultralytics' LetterBox for an input of `shape` (scaleup on)."""
    r = min(imgsz / shape[0], imgsz / shape[1])
    return round(shape[1] * r), round(shape[0] * r)


# ----------------------------------------------------------------------------- detector
class Detector:
    def __init__(self, weights: str, imgsz: int, conf: float = 0.05, half: bool = True, device=None, max_det: int = 5,
                 direct: bool = True):
        from ultralytics import YOLO
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.imgsz, self.conf, self.half, self.device, self.max_det = imgsz, conf, half, device, max_det
        self.backend = None
        if direct and self.weights.endswith(".engine") and self.model.task == "detect":
            self._init_direct()  # segmentation engines keep the ultralytics predictor (mask post-processing)

    # -- TensorRT fast path
    def _init_direct(self):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        dev = self.device
        dev = torch.device("cuda:0") if dev is None else torch.device(f"cuda:{dev}" if str(dev).isdigit() else dev)
        be = AutoBackend(self.weights, device=dev, fp16=self.half, verbose=False)
        b = getattr(be.backend, "bindings", {}).get("images")
        if be.task != "detect" or b is None or getattr(be.backend, "dynamic", True) or b.shape[0] != 1 or b.shape[2] != b.shape[3]:
            return  # dynamic / batched engines keep the ultralytics predictor
        self.backend, self.torch, self.dev = be, torch, dev
        self.inp_size = b.shape[2]
        self.fp16 = bool(be.backend.fp16)  # what ultralytics' preprocess feeds: half only for an fp16 input binding
        self.end2end = bool(getattr(be, "end2end", False))
        self.buf, self.buf_key = None, None

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """The frame resized exactly as ultralytics' LetterBox would before padding (no-op if already that size)."""
        wh = letterbox_unpad(frame.shape, self.imgsz)
        return frame if (frame.shape[1], frame.shape[0]) == wh else cv2.resize(frame, wh, interpolation=cv2.INTER_LINEAR)

    def _direct(self, img):
        from ultralytics.utils import nms, ops
        torch = self.torch
        t0 = time.perf_counter()
        S, shape0 = self.inp_size, img.shape
        h, w = img.shape[:2]
        if letterbox_unpad(img.shape, S) != (w, h):  # tracker imgsz differs from the engine's input size
            img = cv2.resize(img, letterbox_unpad(img.shape, S), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        # LetterBox.get_params with auto=False, center=True (static engine input)
        dw, dh = (S - w) / 2, (S - h) / 2
        top, left = round(dh - 0.1), round(dw - 0.1)
        if self.buf_key != (h, w):  # the padding border never changes, only the image region is rewritten
            pad = torch.full((1, 3, S, S), 114, dtype=torch.uint8, device=self.dev)
            self.buf = pad.half().div_(255) if self.fp16 else pad.float().div_(255)  # same ops as ultralytics' preprocess
            self.buf_key = (h, w)
        x = torch.from_numpy(img).to(self.dev).permute(2, 0, 1).flip(0)  # HWC BGR -> CHW RGB
        self.buf[0, :, top:top + h, left:left + w] = x.half().div_(255) if self.fp16 else x.float().div_(255)
        t1 = time.perf_counter()
        y = self.backend(self.buf)
        # same call as ultralytics' DetectionPredictor.postprocess (iou 0.7 default, class-aware, nc inferred)
        pred = nms.non_max_suppression(y, self.conf, 0.7, None, False, max_det=self.max_det, nc=0, end2end=self.end2end)[0]
        pred[:, :4] = ops.scale_boxes(self.buf.shape[2:], pred[:, :4], shape0)
        p = pred.cpu().numpy()
        t2 = time.perf_counter()
        return p[:, :4], p[:, 4], None, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": 0.0}

    def __call__(self, frame: np.ndarray, small: np.ndarray | None = None):
        """Detections in `frame` pixel coordinates. `small` = self.resize(frame) if the caller already has it."""
        img = self.resize(frame) if small is None else small
        sx, sy = frame.shape[1] / img.shape[1], frame.shape[0] / img.shape[0]
        xyxy = conf = polys = None
        if self.backend is not None:
            xyxy, conf, polys, speed = self._direct(img)
        else:
            kw = {"quantize": 16} if self.half else {}
            r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False, max_det=self.max_det, **kw)[0]
            speed = r.speed
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                polys = r.masks.xy if r.masks is not None else None
        dets = []
        if xyxy is not None and len(conf):
            if (sx, sy) != (1.0, 1.0):
                xyxy = xyxy * np.array([sx, sy, sx, sy], np.float32)
            for i in range(len(conf)):
                cx = cy = np.nan
                if polys is not None and len(polys[i]) >= 3:
                    m = cv2.moments(polys[i].astype(np.float32) * np.array([sx, sy], np.float32))
                    if m["m00"] > 0:
                        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
                dets.append({"box": xyxy[i].astype(float), "conf": float(conf[i]), "seg": (cx, cy)})
        return dets, speed


# ----------------------------------------------------------------------------- background
def _masked_median(frames: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Equals np.nan_to_num(np.nanmedian(stack with masked pixels set to NaN, axis=0), nan=127) as float32, bit for bit
    (inputs are uint8, so the mean of the two middle values is exact). GPU sort when available, else numpy."""
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    n_f, H = len(frames), frames[0].shape[0]
    out = np.empty(frames[0].shape, np.float32)
    step = max(1, H // 4)  # row chunks keep GPU memory ~100 MB
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        if gpu:
            s = torch.from_numpy(np.stack([f[y0:y1] for f in frames])).cuda().float()
            s[torch.from_numpy(np.stack([m[y0:y1] for m in masks])).cuda()] = float("nan")
            s = torch.sort(s, dim=0)[0]  # NaN sorts last
            n = (~torch.isnan(s)).sum(0, keepdim=True)
            med = ((torch.gather(s, 0, (n - 1).clamp_min(0) // 2) + torch.gather(s, 0, (n // 2).clamp_max(n_f - 1))) * 0.5)[0]
            med[n[0] == 0] = 127.0
            out[y0:y1] = med.cpu().numpy()
        else:
            s = np.stack([f[y0:y1] for f in frames], axis=-1).astype(np.float32)  # (h, W, C, n)
            s[np.broadcast_to(np.stack([m[y0:y1] for m in masks], axis=-1)[:, :, None, :], s.shape)] = np.nan
            s.sort(axis=-1)
            n = np.count_nonzero(~np.isnan(s), axis=-1)
            a = np.take_along_axis(s, (np.maximum(n - 1, 0) // 2)[..., None], -1)[..., 0]
            b = np.take_along_axis(s, np.minimum(n // 2, n_f - 1)[..., None], -1)[..., 0]
            med = (a + b) * np.float32(0.5)
            med[n == 0] = 127.0
            out[y0:y1] = med
    return out


class OnlineBackground:
    """Median of the first `warmup` frames (detected drone boxes masked out), then a slow running
    average that never learns pixels inside detected boxes, so a hovering drone is not absorbed.
    threaded=True: update() returns at once and runs in a worker thread; every read (bg, ready, wait())
    first waits for it, so results are identical to the synchronous version."""

    def __init__(self, scale: float = 0.5, warmup: int = 30, alpha: float = 0.02, motion_px: float = 1.0, threaded: bool = True):
        self.scale, self.warmup, self.alpha, self.motion_px = scale, warmup, alpha, motion_px
        self.stack, self.stack_m, self._bg, self.prev = [], [], None, None
        self.moving = False
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="bg") if threaded else None
        self._pending = None

    def small(self, frame):
        return cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def reuse(self, frame, resized) -> bool:
        """True if `resized` (cv2 INTER_LINEAR) equals small(frame): an exact 2x downscale, which OpenCV runs as INTER_AREA."""
        return (self.scale == 0.5 and resized.shape[0] * 2 == frame.shape[0] and resized.shape[1] * 2 == frame.shape[1]
                and frame.dtype == np.uint8)

    def _mask(self, shape, boxes, pad=0.25):
        m = np.zeros(shape[:2], bool)
        for b in boxes:
            x0, y0, x1, y1 = b * self.scale
            w, h = x1 - x0, y1 - y0
            m[max(0, int(y0 - pad * h)):int(y1 + pad * h) + 1, max(0, int(x0 - pad * w)):int(x1 + pad * w) + 1] = True
        return m

    def wait(self):
        if self._pending is not None:
            p, self._pending = self._pending, None
            p.result()

    def update(self, fs, boxes):
        if self._pool is None:
            return self._update(fs, boxes)
        self.wait()
        self._pending = self._pool.submit(self._update, fs, list(boxes))

    def _update(self, fs, boxes):
        g = cv2.cvtColor(cv2.resize(fs, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev is not None:
            shift = float(np.hypot(*cv2.phaseCorrelate(self.prev, g)[0]))
            self.moving = shift > self.motion_px
            if self.moving:
                self.stack, self.stack_m, self._bg = [], [], None
        self.prev = g
        if self.moving:
            return
        m = self._mask(fs.shape, boxes)
        if self._bg is None:
            self.stack.append(fs)
            self.stack_m.append(m)
            if len(self.stack) >= self.warmup:
                self._bg = _masked_median(self.stack, self.stack_m)
                self.stack, self.stack_m = [], []
            return
        cv2.accumulateWeighted(fs, self._bg, self.alpha, mask=(~m).astype(np.uint8))

    @property
    def bg(self):
        self.wait()
        return self._bg

    @property
    def ready(self):
        return self.bg is not None


def roi_refine(fs, bg, box, scale, thr_hi=30, thr_lo=12, pad=0.15, min_fill=0.10):
    """Background subtraction inside a detection box. Adaptive (Otsu, clipped) threshold, all components
    whose centroid lies in the box are merged. Returns (cx, cy, fill) in full-res px or None."""
    H, W = fs.shape[:2]
    x0, y0, x1, y1 = box * scale
    w, h = x1 - x0, y1 - y0
    ax0, ay0 = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    ax1, ay1 = min(W, int(x1 + pad * w) + 1), min(H, int(y1 + pad * h) + 1)
    if ax1 - ax0 < 4 or ay1 - ay0 < 4:
        return None
    a = cv2.GaussianBlur(fs[ay0:ay1, ax0:ax1], (3, 3), 0).astype(np.int16)
    b = cv2.GaussianBlur(bg[ay0:ay1, ax0:ax1].astype(np.uint8), (3, 3), 0).astype(np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)
    t_otsu, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t_otsu, thr_lo, thr_hi))
    bw = (d > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, bool)
    bx0, by0, bx1, by1 = x0 - ax0, y0 - ay0, x1 - ax0, y1 - ay0
    for i in range(1, n):
        cx, cy = cen[i]
        if bx0 <= cx <= bx1 and by0 <= cy <= by1 and st[i, cv2.CC_STAT_AREA] >= 4:
            keep[i] = True
    if not keep.any():
        return None
    m = keep[lab]
    fill = m.sum() / max(1.0, w * h)
    if fill < min_fill:
        return None
    ys, xs = np.nonzero(m)
    return (xs.mean() + ax0) / scale, (ys.mean() + ay0) / scale, float(fill)


def measure(frame, detector: Detector, bgm: OnlineBackground, cfg: dict, topk: int = 3):
    """Detector + ROI refinement for the top-k detections + background update. Returns (dets, timing).
    With a threaded bgm this frame's background update overlaps the next call's detector."""
    t0 = time.perf_counter()
    small = detector.resize(frame)
    dets, speed = detector(frame, small)
    t1 = time.perf_counter()
    fs = small if bgm.reuse(frame, small) else bgm.small(frame)
    max_area = cfg.get("max_box_frac", 0.25) * frame.shape[0] * frame.shape[1]
    bg = bgm.bg  # waits for the previous frame's update
    for d in dets[:topk]:
        b = d["box"]
        if (b[2] - b[0]) * (b[3] - b[1]) > max_area:
            d["fuse"], d["fill"] = (np.nan, np.nan), 0.0
            continue
        r = roi_refine(fs, bg, d["box"], bgm.scale, cfg["thr_hi"], cfg["thr_lo"], cfg["roi_pad"], cfg["min_fill"]) if bg is not None else None
        d["fuse"] = (r[0], r[1]) if r else (np.nan, np.nan)
        d["fill"] = r[2] if r else 0.0
    for d in dets[topk:]:
        d["fuse"], d["fill"] = (np.nan, np.nan), 0.0
    bgm.update(fs, [d["box"] for d in dets if d["conf"] >= cfg["bg_mask_conf"]])
    t2 = time.perf_counter()
    return dets, {"det_ms": (t1 - t0) * 1e3, "roi_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3, **{f"yolo_{k}": v for k, v in speed.items()}}


# ----------------------------------------------------------------------------- association + Kalman
class Kalman:
    def __init__(self, q: float, r: float):
        self.q, self.r = q, r
        self.x = None
        self.P = None
        self.t = None
        self.miss = 0

    def predict(self, t):
        if self.x is None:
            return None
        dt = max(1e-3, t - self.t)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * G @ G.T
        self.t = t
        return self.x[:2]

    def update(self, z, r_scale=1.0):
        if self.x is None:
            return
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = Hm @ self.P @ Hm.T + np.eye(2) * self.r * r_scale
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def init(self, z, t):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.r, self.r, 1e5, 1e5])
        self.t = t
        self.miss = 0


class StaticFilter:
    """Drops detections that sit on a fixed spot: the box centre has stayed within `static_px` of an anchor for at least
    `static_s` seconds. Catches static look-alikes (ceiling light fixtures) that a detector trained without them fires on
    in every frame. A real drone keeps drifting by more than that while hovering; once a suppressed spot moves, its
    anchor resets and it is tracked again. static_s = 0 disables the filter."""

    def __init__(self, static_s: float = 0.0, static_px: float = 1.5, forget_s: float = 1.0):
        self.static_s, self.static_px, self.forget_s = static_s, static_px, forget_s
        self.spots = []  # [anchor_x, anchor_y, first_t, last_t]

    def __call__(self, dets, t):
        if self.static_s <= 0:
            return dets
        self.spots = [s for s in self.spots if t - s[3] <= self.forget_s]
        keep = []
        for d in dets:
            b = d["box"]
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            near = min(self.spots, key=lambda s: np.hypot(c[0] - s[0], c[1] - s[1]), default=None)
            if near is not None and np.hypot(c[0] - near[0], c[1] - near[1]) <= 4 * self.static_px:
                if np.hypot(c[0] - near[0], c[1] - near[1]) > self.static_px:
                    near[:] = [c[0], c[1], t, t]  # it moved: restart the clock at the new spot
                else:
                    near[3] = t
                if near[3] - near[2] >= self.static_s:
                    continue
            else:
                self.spots.append([c[0], c[1], t, t])
            keep.append(d)
        return keep


@dataclass
class Associator:
    """Pick the drone among detections and filter it.
    mode: 'box' (box centre), 'seg' (mask centroid), 'fuse' (ROI-BSM centroid, falls back to seg then box)."""
    mode: str = "fuse"
    conf: float = 0.3
    conf_reacquire: float = 0.5
    gate_px: float = 120.0
    coast: int = 10
    kalman: bool = True
    q: float = 2000000.0
    r: float = 9.0
    static_s: float = 0.0
    static_px: float = 1.5
    kf: Kalman = field(default=None)
    sf: StaticFilter = field(default=None)

    def __post_init__(self):
        self.kf = Kalman(self.q, self.r)
        self.sf = StaticFilter(self.static_s, self.static_px)

    def point(self, d):
        b = d["box"]
        box_c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        if self.mode == "box":
            return box_c
        if self.mode == "seg":
            return d["seg"] if np.isfinite(d["seg"][0]) else box_c
        if np.isfinite(d["fuse"][0]):
            return d["fuse"]
        return d["seg"] if np.isfinite(d["seg"][0]) else box_c

    def step(self, dets, t):
        pred = self.kf.predict(t) if self.kalman else None
        cand = self.sf([d for d in dets if d["conf"] >= self.conf], t)
        chosen = None
        if pred is not None and self.kf.miss <= self.coast:
            near = [d for d in cand if np.hypot(*(np.asarray(self.point(d)) - pred)) <= self.gate_px * (1 + 0.5 * self.kf.miss)]
            if near:
                chosen = max(near, key=lambda d: d["conf"])
            else:
                far = [d for d in cand if d["conf"] >= self.conf_reacquire]
                chosen = max(far, key=lambda d: d["conf"]) if far else None
        elif cand:
            chosen = max(cand, key=lambda d: d["conf"])
        if chosen is None:
            if self.kalman and self.kf.x is not None:
                self.kf.miss += 1
                if self.kf.miss <= self.coast:
                    return {"x": float(self.kf.x[0]), "y": float(self.kf.x[1]), "det": None, "coast": True}
                self.kf.x = None
            return {"x": np.nan, "y": np.nan, "det": None, "coast": False}
        z = self.point(chosen)
        if not self.kalman:
            return {"x": float(z[0]), "y": float(z[1]), "det": chosen, "coast": False}
        if self.kf.x is None or self.kf.miss > self.coast or np.hypot(*(np.asarray(z) - pred)) > 3 * self.gate_px:
            self.kf.init(z, t)
        else:
            self.kf.update(z, 1.0 / max(chosen["conf"], 0.1))
            self.kf.miss = 0
        return {"x": float(self.kf.x[0]), "y": float(self.kf.x[1]), "det": chosen, "coast": False}
