"""Frame sources with a common interface: read() -> (ok, frame, t_seconds).

Camera     : USB webcam in a grabber thread; read() returns the newest frame only (no queue latency).
VideoFile  : recorded video, timestamps from the frame index (offline tests / replays).
SimCamera  : renders a SimTello (and an optional simulated laser dot) onto a real background photo, so the full
             vision + control + logging chain runs without hardware.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Camera:
    def __init__(self, index=0, width=1920, height=1080, fps=30, fourcc="MJPG", backend="dshow", record: str | None = None):
        api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}.get(backend, cv2.CAP_ANY)
        self.cap = cv2.VideoCapture(index, api)
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} not available")
        self.t0 = time.perf_counter()
        self.frame, self.t, self.n = None, 0.0, 0
        self.cond = threading.Condition()
        self.alive = True
        self.writer = None
        self.record = record
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps_nominal = self.cap.get(cv2.CAP_PROP_FPS)
        threading.Thread(target=self._grab, daemon=True).start()
        self._last_n = 0

    def _grab(self):
        while self.alive:
            ok, f = self.cap.read()
            t = time.perf_counter() - self.t0
            if not ok:
                time.sleep(0.005)
                continue
            if self.record:
                if self.writer is None:
                    self.writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), max(10.0, self.fps_nominal or 30), (f.shape[1], f.shape[0]))
                self.writer.write(f)
            with self.cond:
                self.frame, self.t, self.n = f, t, self.n + 1
                self.cond.notify_all()

    def read(self, timeout=1.0):
        """Block until a frame newer than the last one returned is available."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.n > self._last_n, timeout):
                return False, None, None
            self._last_n = self.n
            return True, self.frame, self.t

    def now(self):
        return time.perf_counter() - self.t0

    def release(self):
        self.alive = False
        time.sleep(0.1)
        self.cap.release()
        if self.writer is not None:
            self.writer.release()


class VideoFile:
    def __init__(self, path, realtime=False):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.i = 0
        self.realtime = realtime
        self.t0 = time.perf_counter()

    def read(self, timeout=1.0):
        ok, f = self.cap.read()
        if not ok:
            return False, None, None
        t = self.i / self.fps
        self.i += 1
        if self.realtime:
            d = t - (time.perf_counter() - self.t0)
            if d > 0:
                time.sleep(d)
        return True, f, t

    def now(self):
        return self.i / self.fps

    def release(self):
        self.cap.release()


class SimCamera:
    """Renders the simulated drone at its wall position. The plant advances by 1/fps per frame, so a simulated
    flight is deterministic and can run faster than real time (realtime=False)."""

    def __init__(self, background, scene, drone, fps=25.0, realtime=False, noise=2.0, laser=None, assets=None):
        self.bg = background
        self.sc = scene
        self.drone = drone
        self.fps = fps
        self.dt = 1.0 / fps
        self.t = 0.0
        self.realtime = realtime
        self.t0 = time.perf_counter()
        self.noise = noise
        self.laser = laser  # callable t -> wall (X, Y) or None
        assets = Path(assets or Path(__file__).resolve().parent.parent / "assets")
        spr = cv2.imread(str(assets / "drone_sprite.png"), cv2.IMREAD_UNCHANGED)
        meta = json.loads((assets / "drone_sprite.json").read_text())
        k = meta["cm_per_px"] / scene.s_px  # sprite was cut at another zoom: rescale to this scene
        self.spr = cv2.resize(spr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        self.rng = np.random.default_rng(1)

    def read(self, timeout=1.0):
        self.drone.step(self.dt)
        self.t += self.dt
        if self.realtime:
            d = self.t - (time.perf_counter() - self.t0)
            if d > 0:
                time.sleep(d)
        f = self.bg.copy()
        uv = self.sc.to_px(self.drone.p)
        _paste(f, self.spr, uv)
        if self.laser is not None:
            L = self.laser(self.t)
            if L is not None:
                c = tuple(np.int32(self.sc.to_px(L)))
                cv2.ellipse(f, c, (9, 5), 30, 0, 360, (90, 255, 120), -1)
                cv2.ellipse(f, c, (5, 3), 30, 0, 360, (245, 255, 245), -1)
        if self.noise:
            if not hasattr(self, "_nz"):
                self._nz = [self.rng.normal(0, self.noise, f.shape).astype(np.int16) for _ in range(3)]
            f = np.clip(f.astype(np.int16) + self._nz[int(self.t * self.fps) % 3], 0, 255).astype(np.uint8)
        return True, f, self.t

    def now(self):
        return self.t

    def release(self):
        pass


def _paste(img, spr, centre):
    h, w = spr.shape[:2]
    x0, y0 = int(round(centre[0] - w / 2)), int(round(centre[1] - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4].astype(np.float32) / 255
    img[ya:yb, xa:xb] = (s[..., :3] * a + img[ya:yb, xa:xb] * (1 - a)).astype(np.uint8)
