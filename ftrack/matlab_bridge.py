"""Bridge called from MATLAB (pyenv OutOfProcess): camera capture and, optionally, YOLO26 drone + laser detection.

MATLAB stays the main program (scene calibration, control, Tello, safety, logging, dashboard). This module only
provides what MATLAB cannot do here: an OpenCV camera when the MATLAB webcam support package is missing, and the
YOLO detector (no Deep Learning Toolbox). Frames go to MATLAB as RGB numpy arrays (half or quarter resolution, a
full frame only for the one-time calibration) because every MATLAB<->Python image transfer costs ~15 ms at 960x540.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera

ROOT = Path(__file__).resolve().parent.parent
_S: dict = {}


def open_camera(index, width=1920, height=1080, fps=30, record=""):
    """index: int or 'auto' (first camera 0..4 in which any board calibrates). Returns (index, w, h)."""
    from .scene import calibrate
    if str(index) == "auto":
        idx = 0
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            f = None
            for _ in range(15):
                ok, fr = cap.read()
                f = fr if ok else f
            cap.release()
            if f is None:
                continue
            found = False
            for k in ("circle", "square", "path"):
                try:
                    calibrate(f, k)
                    found = True
                    break
                except Exception:
                    pass
            if found:
                idx = i
                break
        index = idx
    cam = Camera(int(index), int(width), int(height), int(fps), "MJPG", "dshow", record=str(record) or None)
    _S["cam"] = cam
    return int(index), int(cam.size[0]), int(cam.size[1])


def full_frame():
    """Newest full-resolution frame, RGB (for the one-time scene calibration)."""
    ok, f, t = _S["cam"].read(timeout=2.0)
    if not ok:
        raise RuntimeError("camera returned no frame")
    return np.ascontiguousarray(f[:, :, ::-1])


def half_frame():
    """(t, RGB half-resolution frame) of the next new camera frame (MATLAB background-subtraction mode)."""
    ok, f, t = _S["cam"].read(timeout=2.0)
    if not ok:
        return -1.0, np.zeros((2, 2, 3), np.uint8)
    h = cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(t), np.ascontiguousarray(h[:, :, ::-1])


def setup_vision(work_poly_px, cm_per_px, detector="yolo", model_dir="models/final_det", laser=False):
    """work_poly_px: Nx2 (full-res px) polygon of the work area, from MATLAB's calibration."""
    from .vision import DroneVision, LaserDetector
    poly = np.asarray(work_poly_px, float).reshape(-1, 2)
    cfg = {"detector": detector, "model_dir": model_dir, "conf": 0.25, "coast": 3, "min_box_cm": 6, "max_box_cm": 60}
    v = DroneVision(cfg, None, ROOT)
    v.work_poly, v.cmpp = poly, float(cm_per_px)
    v.min_box_px, v.max_box_px = 6 / v.cmpp, 60 / v.cmpp
    _S["vision"] = v
    _S["laser"] = LaserDetector(poly, float(cm_per_px)) if laser else None
    return v.kind


def detect(preview_scale=0.25, want_laser=False):
    """Next camera frame -> drone (+ laser) measurement, all in full-resolution px, plus an RGB preview.
    Returns (vector, preview): vector = [t, u, v, valid, conf, x0, y0, x1, y1, vu, vv, vision_ms, lu, lv]."""
    ok, f, t = _S["cam"].read(timeout=2.0)
    if not ok:
        return np.full(14, np.nan), np.zeros((2, 2, 3), np.uint8)
    o = _S["vision"](f, t)
    L = _S["laser"](f) if (want_laser and _S.get("laser") is not None) else None
    uv = o["uv"]
    box = o["box"] if o["box"] is not None else (np.nan,) * 4
    vel = o["vel_px"] if o["vel_px"] is not None else (0.0, 0.0)
    vec = np.array([t, *(uv if uv is not None else (np.nan, np.nan)), float(uv is not None), o["conf"], *box, *vel, o["ms"],
                    *(L if L is not None else (np.nan, np.nan))], float)
    p = cv2.resize(f, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
    return vec, np.ascontiguousarray(p[:, :, ::-1])


def close():
    cam = _S.pop("cam", None)
    if cam is not None:
        cam.release()
    return True
