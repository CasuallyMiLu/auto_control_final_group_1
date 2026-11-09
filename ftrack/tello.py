"""Minimal DJI/Ryze Tello (EDU) SDK 2.0/3.0 client over UDP, plus a simulated drone with the same interface.

Real drone: connect the laptop Wi-Fi to TELLO-XXXXXX, then Tello(); everything else is plain UDP.
  - commands  -> 192.168.10.1:8889, replies come back to our local port 8889
  - state     <- broadcast on local port 8890 (~10 Hz): pitch, roll, yaw, vgx, vgy, vgz, tof, h, bat, ...
rc a b c d (each -100..100): a = left/right (roll), b = forward/back (pitch), c = up/down, d = yaw.
The drone auto-lands if it receives nothing for 15 s; the control loops send rc every frame.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import deque

import numpy as np


class Tello:
    def __init__(self, ip: str = "192.168.10.1", cmd_port: int = 8889, state_port: int = 8890, timeout: float = 7.0):
        self.addr = (ip, cmd_port)
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", cmd_port))
        self.sock.settimeout(0.2)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("", state_port))
        self.state_sock.settimeout(0.5)
        self.state: dict = {}
        self.state_t = 0.0
        self._replies = deque()
        self._alive = True
        self._lock = threading.Lock()
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._rx_state, daemon=True).start()
        self.flying = False
        self.last_rc = (0, 0, 0, 0)

    # ---------------------------------------------------------------- receive threads
    def _rx(self):
        while self._alive:
            try:
                data, _ = self.sock.recvfrom(1024)
                self._replies.append(data.decode(errors="ignore").strip())
            except (socket.timeout, OSError):
                continue

    def _rx_state(self):
        while self._alive:
            try:
                data, _ = self.state_sock.recvfrom(1024)
            except (socket.timeout, OSError):
                continue
            st = {}
            for kv in data.decode(errors="ignore").strip().split(";"):
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    try:
                        st[k] = float(v)
                    except ValueError:
                        st[k] = v
            self.state, self.state_t = st, time.time()

    # ---------------------------------------------------------------- commands
    def send(self, cmd: str, wait: bool = True, timeout: float | None = None) -> str | None:
        with self._lock:
            self._replies.clear()
            self.sock.sendto(cmd.encode(), self.addr)
            if not wait:
                return None
            t_end = time.time() + (timeout or self.timeout)
            while time.time() < t_end:
                if self._replies:
                    return self._replies.popleft()
                time.sleep(0.01)
        return None

    def connect(self, retries: int = 5) -> bool:
        for _ in range(retries):
            r = self.send("command", timeout=2.0)
            if r and r.lower().startswith("ok"):
                return True
            time.sleep(0.5)
        return False

    def battery(self) -> float:
        if "bat" in self.state and time.time() - self.state_t < 2:
            return float(self.state["bat"])
        r = self.send("battery?", timeout=2.0)
        try:
            return float(r)
        except (TypeError, ValueError):
            return float("nan")

    def takeoff(self) -> bool:
        r = self.send("takeoff", timeout=12.0)
        self.flying = True  # even without a reply it may be in the air: treat as flying so land() is sent
        return bool(r) and r.lower().startswith("ok")

    def land(self) -> bool:
        self.rc(0, 0, 0, 0)
        r = self.send("land", timeout=8.0)
        self.flying = False
        return bool(r) and r.lower().startswith("ok")

    def emergency(self):
        self.send("emergency", wait=False)
        self.flying = False

    def rc(self, a: float, b: float, c: float, d: float):
        a, b, c, d = (int(np.clip(round(x), -100, 100)) for x in (a, b, c, d))
        self.last_rc = (a, b, c, d)
        self.sock.sendto(f"rc {a} {b} {c} {d}".encode(), self.addr)

    def close(self):
        self._alive = False
        for s in (self.sock, self.state_sock):
            try:
                s.close()
            except OSError:
                pass


class SimTello:
    """Same interface as Tello, for dry runs. Wall-plane plant (same structure the controller is designed for):
        v_ss = B @ [rc_a, rc_c]   (cm/s),   tau_i dv_i/dt = v_ss,i(t - L) - v_i,   dp/dt = v + wind
    model.json supplies B, tau, L. Wind: slow random walk. Take-off climbs ~80 cm, land descends."""

    def __init__(self, model: dict, start_xy=(0.0, 0.0), seed: int = 0, wind_cm_s: float = 3.0, takeoff_height_cm: float = 80.0):
        self.B = np.asarray(model["B"], float)
        self.tau = np.array([model["axes"]["x"]["tau"], model["axes"]["y"]["tau"]])
        self.L = float(model["axes"]["x"]["L"])
        self.p = np.array(start_xy, float)
        self.ground_y = float(start_xy[1])
        self.v = np.zeros(2)
        self.takeoff_h = takeoff_height_cm
        self.rng = np.random.default_rng(seed)
        self.wind = np.zeros(2)
        self.wind_sigma = wind_cm_s
        self.queue = deque()
        self.u = np.zeros(2)
        self.t = 0.0
        self.flying = False
        self.state = {"bat": 87.0, "h": 0.0, "tof": 10.0}
        self.last_rc = (0, 0, 0, 0)
        self._climb = 0.0

    def connect(self, retries: int = 1) -> bool:
        return True

    def battery(self) -> float:
        return self.state["bat"]

    def takeoff(self) -> bool:
        self.flying = True
        self._climb = self.takeoff_h
        time.sleep(0.05)
        return True

    def land(self) -> bool:
        self.flying = False
        self.u[:] = 0
        self.queue.clear()
        return True

    def emergency(self):
        self.land()

    def rc(self, a, b, c, d):
        a, b, c, d = (int(np.clip(round(x), -100, 100)) for x in (a, b, c, d))
        self.last_rc = (a, b, c, d)
        self.queue.append((self.t + self.L, np.array([a, c], float)))

    def step(self, dt: float):
        self.t += dt
        while self.queue and self.queue[0][0] <= self.t:
            self.u = self.queue.popleft()[1]
        if not self.flying:
            self.p[1] = max(self.ground_y, self.p[1] - 50 * dt)
            self.v[:] = 0
            return
        if self._climb > 0:
            dz = min(self._climb, 60 * dt)
            self.p[1] += dz
            self._climb -= dz
            return
        self.wind += self.rng.normal(0, self.wind_sigma * np.sqrt(dt), 2) - 0.3 * self.wind * dt
        self.v += (self.B @ self.u - self.v) / self.tau * dt
        self.p += (self.v + self.wind) * dt
        self.state["bat"] = max(0.0, self.state["bat"] - dt * 0.02)
        self.state["h"] = self.p[1] - self.ground_y

    def close(self):
        pass
