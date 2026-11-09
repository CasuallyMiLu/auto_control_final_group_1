"""Mission runner shared by theme1.py, theme2.py and sysid_flight.py: one program from start-up to landing.

  calibrate the scene on the live frame -> preview (press G to go) -> connect + battery check -> take-off
  -> acquire the drone in the image -> orientation probe (short climb + short sideways nudge: confirms which way is
  up and the sign of rc 'a') -> mission phases (subclass) -> land -> save log, meta, analysis.

Every camera frame: vision -> wall position -> phase controller -> rc command -> safety -> log -> dashboard.
Keys (dashboard window): SPACE or q = land now, p = pause (hover) / resume, ESC = emergency motor stop.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, SimCamera
from .control import Actuator
from .scene import Geometry, calibrate, orient_closed_path
from .tello import SimTello, Tello
from .viz import Dashboard
from .vision import DroneVision

ROOT = Path(__file__).resolve().parent.parent

LOG_COLS = ["t", "frame", "phase", "valid", "u", "v", "X", "Y", "vX", "vY", "s", "e_n", "refX", "refY", "v_ff",
            "vdesX", "vdesY", "rc_a", "rc_b", "rc_c", "rc_d", "laser_u", "laser_v", "laserX", "laserY",
            "box_w", "box_h", "conf", "vision_ms", "loop_ms", "bat", "h", "tof", "vgx", "vgy", "vgz", "pitch", "roll", "yaw",
            "progress", "excite_axis", "excite_u"]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def base_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--sim", action="store_true", help="simulated drone + rendered camera (no hardware)")
    ap.add_argument("--realtime", action="store_true", help="sim: run at real speed")
    ap.add_argument("--yes", action="store_true", help="skip the preview / go confirmation")
    ap.add_argument("--no-show", action="store_true", help="no window (sim batch tests)")
    ap.add_argument("--rig", default=str(ROOT / "config/rig.json"))
    ap.add_argument("--model", default=str(ROOT / "config/model.json"))
    ap.add_argument("--gains", default=str(ROOT / "config/gains.json"))
    ap.add_argument("--detector", default=None, help="yolo | bsm (override rig.json)")
    ap.add_argument("--probe", default="auto", help="auto | off  (orientation probe after take-off)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--up-hint", default=None, help="override rig.json up_hint: auto|up|right|down|left")
    ap.add_argument("--set", action="append", default=[], help="override a gain, e.g. --set path.v_max=30 --set cross.x.Kp=2")
    ap.add_argument("--sim-model", default=None, help="sim: plant model json for the simulated drone (default: --model)")
    ap.add_argument("--sim-wind", type=float, default=3.0, help="sim: random-walk disturbance strength (cm/s/sqrt(s))")
    ap.add_argument("--sim-flip-a", action="store_true", help="sim: drone faces the camera (rc a reversed) to test the probe")
    return ap


def apply_overrides(d: dict, sets):
    for s in sets:
        k, v = s.split("=", 1)
        ks = k.split(".")
        node = d
        for kk in ks[:-1]:
            node = node[kk]
        node[ks[-1]] = json.loads(v)
    return d


def start_angle(cp: dict, kind: str):
    a = cp.get("start_angle_deg")
    return a.get(kind) if isinstance(a, dict) else a


class Mission:
    name = "mission"
    kind = "circle"

    def __init__(self, args):
        self.a = args
        self.rig = load_json(args.rig)
        self.model = load_json(args.model)
        self.gains = apply_overrides(load_json(args.gains), args.set)
        if args.detector:
            self.rig["vision"]["detector"] = args.detector
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ROOT / "runs" / f"{stamp}_{self.name}{'_sim' if args.sim else ''}{('_' + args.tag) if args.tag else ''}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.phase = "INIT"
        self.phase_t0 = 0.0
        self.t = 0.0
        self.frame_i = 0
        self.p = None  # wall position (cm) of the drone, None if not seen
        self.v_est = np.zeros(2)
        self.last_seen_t = -1e9
        self.paused = False
        self.events = []
        self.msg = ""
        self.st_extra = {}
        self.flight_t0 = None
        self.land_reason = None
        self.takeoff_thread = None
        self.land_thread = None
        self.abort = False
        self.track_err = []
        self.probe_info = {}

    # ------------------------------------------------------------------------------------------ setup
    def log_event(self, s):
        print(f"[{self.t:7.2f}] {s}", flush=True)
        self.events.append({"t": round(self.t, 3), "event": s})
        self.msg = s

    def make_source(self):
        g = Geometry(**self.rig["geometry"])
        cp = self.rig.get("closed_path", {})
        if self.a.sim:
            bg = cv2.imread(str(ROOT / "assets" / f"background_{self.kind}.jpg"))
            self.sc_true = calibrate(bg, self.kind, g, self.rig.get("up_hint", "auto"), cp.get("direction", "ccw"), start_angle(cp, self.kind))
            simm = load_json(self.a.sim_model) if self.a.sim_model else self.model
            simm = copy.deepcopy(simm)
            if self.a.sim_flip_a:
                simm["B"] = (np.array(simm["B"]) * np.array([[-1, 1], [-1, 1]])).tolist()
            self.sim_drone = SimTello(simm, start_xy=self.sim_start_xy(self.sc_true), wind_cm_s=self.a.sim_wind)
            self.src = SimCamera(bg, self.sc_true, self.sim_drone, fps=self.rig["camera"].get("fps", 30) * 0 + 25.0,
                                 realtime=self.a.realtime, laser=self.sim_laser())
            self.tello = self.sim_drone
        else:
            c = self.rig["camera"]
            idx = c["index"]
            if str(idx) == "auto":
                idx = self.find_camera(c)
            self.src = Camera(int(idx), c["width"], c["height"], c.get("fps", 30), c.get("fourcc", "MJPG"), c.get("backend", "dshow"),
                              record=str(self.run_dir / "camera_raw.mp4") if c.get("record_raw", True) else None)
            self.log_event(f"camera {idx}: {self.src.size[0]}x{self.src.size[1]}")
            self.tello = None  # connected after the preview

    def find_camera(self, c):
        """camera.index = "auto": the first camera (0..4) in whose image the board of this scene calibrates."""
        g = Geometry(**self.rig["geometry"])
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
            f = None
            for _ in range(15):
                ok, fr = cap.read()
                f = fr if ok else f
            cap.release()
            if f is None:
                continue
            try:
                calibrate(f, self.kind, g)
                print(f"[camera] index {i}: board found", flush=True)
                return i
            except Exception as e:
                print(f"[camera] index {i}: no board ({e})", flush=True)
        print("[camera] no camera sees the board; using index 0", flush=True)
        return 0

    def sim_start_xy(self, sc):
        return (0.0, 0.0)

    def sim_laser(self):
        return None

    def calibrate_scene(self, timeout=15.0):
        g = Geometry(**self.rig["geometry"])
        up_hint = self.a.up_hint or self.rig.get("up_hint", "auto")
        cp = self.rig.get("closed_path", {})
        t_end = time.time() + timeout
        last_err = None
        frames = []
        while time.time() < t_end:
            ok, f, t = self.src.read()
            if not ok:
                continue
            frames.append(f)
            if len(frames) < 5:
                continue
            try:
                sc = calibrate(frames[-1], self.kind, g, up_hint, cp.get("direction", "ccw"), start_angle(cp, self.kind))
                self.first_frame = frames[-1]
                return sc
            except Exception as e:  # keep trying (someone in front of the camera, exposure settling, ...)
                last_err = e
                frames = frames[-2:]
        raise RuntimeError(f"scene calibration failed: {last_err}")

    def setup(self):
        self.make_source()
        self.sc = self.calibrate_scene()
        info = {k: v for k, v in self.sc.info.items() if k != "homography"}
        self.log_event(f"scene {self.kind}: {self.sc.s_px:.4f} cm/px, path {self.sc.path.length:.0f} cm, up={self.sc.up.round(2).tolist()} ({info.get('up_source')})")
        self.vision = DroneVision(self.rig["vision"], self.sc, ROOT)
        self.log_event(f"vision: {self.vision.kind}" + (f" ({Path(self.vision.weights).name})" if self.vision.kind == "yolo" else ""))
        self.act = Actuator(self.model["B"], **self.gains["actuator"])
        self.dash = Dashboard(self.sc, title=self.title(), record=str(self.run_dir / "dashboard.mp4"), show=not self.a.no_show)
        self.logf = open(self.run_dir / "log.csv", "w", newline="")
        self.logw = csv.writer(self.logf)
        self.logw.writerow(LOG_COLS)
        self.lo, self.hi = self.sc.bounds_cm(self.rig["safety"]["bounds_margin_cm"])
        self.build()

    def title(self):
        return self.name

    def build(self):
        """Subclass: create controllers for the scene."""

    # ------------------------------------------------------------------------------------------ main
    def run(self):
        try:
            self.setup()
            if not self.preview():
                self.log_event("aborted at preview")
                return
            self.connect()
            self.set_phase("TAKEOFF")
            self.flight_t0 = self.t
            self.takeoff_thread = threading.Thread(target=self._takeoff, daemon=True)
            self.takeoff_thread.start()
            self.loop()
        except KeyboardInterrupt:
            self.log_event("Ctrl-C")
        except Exception as e:
            self.log_event(f"ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            self.shutdown()

    def preview(self):
        if self.a.yes or self.a.no_show:
            return True
        self.msg = "check the green path + arrow (start, direction) and the axes. G = go, Q = quit"
        while True:
            ok, f, t = self.src.read()
            if not ok:
                continue
            self.t = t
            o = self.vision(f, t)  # also warms the background model
            uv = o["uv"]
            k = self.dash.render(f, {"phase": "PREVIEW", "t": t, "uv": uv, "box": o["box"], "msg": self.msg,
                                     "xy": self.sc.to_cm(uv) if uv is not None else (np.nan, np.nan)})
            if k in (ord("g"), ord("G"), 13):
                return True
            if k in (ord("q"), ord("Q"), 27):
                return False

    def connect(self):
        if self.a.sim:
            return
        tc = self.rig["tello"]
        self.tello = Tello(tc.get("ip", "192.168.10.1"))
        if not self.tello.connect():
            raise RuntimeError("Tello did not answer 'command' (Wi-Fi connected to TELLO-xxxx? firewall allows python UDP 8889/8890?)")
        time.sleep(0.5)
        bat = self.tello.battery()
        self.log_event(f"Tello connected, battery {bat:.0f}%")
        if bat < tc.get("min_battery", 30):
            raise RuntimeError(f"battery {bat:.0f}% < {tc.get('min_battery', 30)}%: change the battery")

    def _takeoff(self):
        ok = self.tello.takeoff()
        self.takeoff_ok = ok

    def set_phase(self, ph):
        self.phase = ph
        self.phase_t0 = self.t
        self.log_event(f"phase -> {ph}")
        if ph.startswith("PROBE"):
            self.probe_p0 = None
            if hasattr(self, "probe_dir"):
                del self.probe_dir
        self.on_phase(ph)

    def on_phase(self, ph):
        pass

    def loop(self):
        sf = self.rig["safety"]
        t_prev = None
        rate = []
        while True:
            ok, f, t = self.src.read()
            if not ok:
                if self.a.sim:
                    break
                continue
            tl0 = time.perf_counter()
            self.t = t
            dt = 1 / 30 if t_prev is None else max(1e-3, t - t_prev)
            t_prev = t
            self.frame_i += 1
            o = self.vision(f, t)
            self.o = o
            if o["uv"] is not None:
                self.p = self.sc.to_cm(o["uv"])
                if o["vel_px"] is not None:
                    self.v_est = o["vel_px"] @ self.sc.A.T
                self.last_seen_t = t
                self.seen = True
            else:
                self.seen = False
            lost_for = t - self.last_seen_t
            v_des, rc, info = None, (0, 0, 0, 0), {}

            # ---------------------------------------------------------------- phase logic
            if self.phase == "TAKEOFF":
                if not self.takeoff_thread.is_alive():
                    self.set_phase("ACQUIRE")
                elif t - self.phase_t0 > sf["takeoff_timeout_s"]:
                    self.land("take-off timeout")
            elif self.phase == "ACQUIRE":
                rc = (0, 0, 0, 0)
                if self.seen and self.inside(self.p):
                    self.acq = getattr(self, "acq", 0) + 1
                else:
                    self.acq = 0
                if self.acq >= 5:
                    self.set_phase("PROBE_UP" if self.need_probe() else self.first_phase())
                elif t - self.phase_t0 > sf["acquire_timeout_s"]:
                    self.land("drone not found in the image after take-off")
            elif self.phase.startswith("PROBE"):
                rc = self.probe_step()
            elif self.phase in ("LAND", "DONE"):
                rc = None
            elif self.paused:
                rc = (0, 0, 0, 0)
                self.msg = "PAUSED (p to resume, SPACE to land)"
            else:
                if lost_for > sf["lost_land_s"]:
                    self.land(f"drone lost for {lost_for:.1f} s")
                    rc = None
                elif lost_for > sf["lost_hover_s"] or self.p is None:
                    rc = (0, 0, 0, 0)
                    self.act.reset()
                    self.msg = f"drone not visible ({lost_for:.1f} s): hovering"
                else:
                    if not self.inside(self.p):
                        self.land(f"left the safe area at X={self.p[0]:.0f}, Y={self.p[1]:.0f} cm")
                        rc = None
                    else:
                        pp = self.p + self.v_est * self.gains.get("predict_s", 0.0)
                        v_des, info = self.control(pp, dt)
                        if v_des is None:
                            rc = info.pop("rc", (0, 0, 0, 0))
                        else:
                            rc, v_des = self.act(v_des, dt)
            if self.flight_t0 is not None and self.phase not in ("LAND", "DONE") and t - self.flight_t0 > sf["max_flight_s"]:
                self.land("max flight time")
                rc = None
            if rc is not None and self.tello is not None and self.phase not in ("TAKEOFF", "LAND", "DONE"):
                self.tello.rc(*rc)
            self.rc_sent = rc if rc is not None else (0, 0, 0, 0)

            # ---------------------------------------------------------------- log + display
            loop_ms = (time.perf_counter() - tl0) * 1e3
            rate.append(t)
            rate = rate[-30:]
            fps = (len(rate) - 1) / (rate[-1] - rate[0]) if len(rate) > 2 and rate[-1] > rate[0] else 0
            self.write_log(o, v_des, info, loop_ms)
            st = self.dash_state(o, info, fps)
            k = self.dash.render(f, st)
            self.handle_key(k)
            if self.phase == "DONE" and t - self.phase_t0 > 1.0:
                break
            if self.phase == "LAND" and self.land_thread is not None and not self.land_thread.is_alive():
                self.set_phase("DONE")

    def inside(self, p):
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    # ------------------------------------------------------------------------------------------ orientation probe
    def need_probe(self):
        if self.a.probe == "off":
            return False
        return True

    def first_phase(self):
        raise NotImplementedError

    def probe_step(self):
        """PROBE_UP : rc c=+P for T, then still 0.6 s -> image displacement = physical up (fixes symmetric scenes)
           PROBE_SIDE: rc a=+P for T, then still 0.6 s, then back -> sign of rc a along world X (drone facing)."""
        P, T, still = 40, 0.7, 0.6
        t = self.t - self.phase_t0
        if self.probe_p0 is None and self.seen:
            self.probe_p0 = (self.o["uv"].copy(), self.p.copy())
        if self.phase == "PROBE_UP":
            if t < T:
                return (0, 0, P, 0)
            if t < T + still:
                return (0, 0, 0, 0)
            if self.seen and self.probe_p0 is not None:
                duv = self.o["uv"] - self.probe_p0[0]
                if np.linalg.norm(duv) > 8 / self.sc.s_px:
                    up_meas = duv / np.linalg.norm(duv)
                    ang = math.degrees(math.acos(np.clip(np.dot(up_meas, self.sc.up), -1, 1)))
                    self.probe_info["up_measured_img"] = up_meas.round(3).tolist()
                    self.probe_info["up_angle_err_deg"] = round(ang, 1)
                    if ang > 45:
                        changed = self.sc.set_up_vector(up_meas)
                        if changed and self.sc.path.closed:
                            cp = self.rig.get("closed_path", {})
                            orient_closed_path(self.sc, cp.get("direction", "ccw"), start_angle(cp, self.kind))
                        self.log_event(f"probe: up re-oriented by {self.sc.info.get('up_changed_by_deg')} deg")
                        self.lo, self.hi = self.sc.bounds_cm(self.rig["safety"]["bounds_margin_cm"])
                        self.p = self.sc.to_cm(self.o["uv"])
                        self.build()
                    else:
                        self.log_event(f"probe: up confirmed ({ang:.0f} deg off)")
                else:
                    self.log_event("probe: climb too small to judge, keeping the calibrated orientation")
            self.set_phase("PROBE_SIDE")
            return (0, 0, 0, 0)
        if self.phase == "PROBE_SIDE":
            if not hasattr(self, "probe_dir") or t < 0.05:
                # push towards the track centre first (staying in the camera view), then back
                cx = 0.5 * (self.sc.path.pts[:, 0].min() + self.sc.path.pts[:, 0].max())
                px = self.p[0] if self.p is not None else cx
                self.probe_dir = -1.0 if px > cx else 1.0
            sP = self.probe_dir * P
            if t < T:
                return (sP, 0, 0, 0)
            if t < T + still:
                return (0, 0, 0, 0)
            if t < 2 * T + still:
                return (-sP, 0, 0, 0)
            if t < 2 * T + 2 * still:
                return (0, 0, 0, 0)
            # displacement at the end of the first push (logged), judged from the log buffer
            d = self.probe_side_dx()
            if d == "lost":  # an inward push cannot leave the view: it went outward, so rc 'a' is reversed
                self.log_event("probe: drone left the view during the inward push")
                d = -10.0 * self.probe_dir
            d = None if d is None else d * self.probe_dir  # displacement along the commanded (+a) direction
            if d is not None and abs(d) > 4:
                self.probe_info["side_dx_cm"] = round(float(d), 1)
                if d < 0:
                    B = np.array(self.act.B)
                    B[:, 0] *= -1
                    self.act = Actuator(B, **self.gains["actuator"])
                    self.log_event("probe: rc 'a' moves the drone to the LEFT -> drone faces the camera; sign flipped")
                else:
                    self.log_event("probe: rc 'a' sign confirmed")
            else:
                self.log_event("probe: sideways motion too small to judge, keeping rc 'a' sign")
            self.set_phase(self.first_phase())
            return (0, 0, 0, 0)
        return (0, 0, 0, 0)

    def probe_side_dx(self):
        """X displacement (cm) between the start of the push and the last sample seen 0.4..1.3 s after it.
        Returns 'lost' if the drone disappeared after the push started (it went out of the camera view)."""
        if not hasattr(self, "_probe_hist"):
            return None
        h = self._probe_hist
        t0 = self.phase_t0
        a = [p for (t, p) in h if t0 - 0.1 <= t <= t0 + 0.1]
        b = [p for (t, p) in h if t0 + 0.4 <= t <= t0 + 1.3]
        if not a:
            return None
        if not b:
            return "lost"
        return float(b[-1][0] - np.mean(a, axis=0)[0])

    # ------------------------------------------------------------------------------------------ control (subclass)
    def control(self, p, dt):
        return np.zeros(2), {}

    # ------------------------------------------------------------------------------------------ landing / keys
    def land(self, reason):
        if self.phase in ("LAND", "DONE"):
            return
        self.land_reason = reason
        self.log_event(f"LAND: {reason}")
        self.set_phase("LAND")
        if self.tello is not None:
            self.land_thread = threading.Thread(target=self.tello.land, daemon=True)
            self.land_thread.start()

    def handle_key(self, k):
        if k < 0 or k == 255:
            return
        if k in (ord(" "), ord("q"), ord("Q")):
            self.land("key")
        elif k == 27:
            self.log_event("EMERGENCY STOP (ESC)")
            if self.tello is not None:
                self.tello.emergency()
            self.set_phase("DONE")
        elif k in (ord("l"), ord("L")):
            self.land("L key")
        elif k in (ord("p"), ord("P")):
            self.paused = not self.paused
            self.log_event("pause" if self.paused else "resume")

    # ------------------------------------------------------------------------------------------ log / dash
    def write_log(self, o, v_des, info, loop_ms):
        st = getattr(self.tello, "state", {}) or {}
        p = self.p if self.seen else (np.nan, np.nan)
        if self.seen:
            if not hasattr(self, "_probe_hist"):
                self._probe_hist = []
            self._probe_hist.append((self.t, self.p.copy()))
            self._probe_hist = self._probe_hist[-200:]
        uv = o["uv"] if o["uv"] is not None else (np.nan, np.nan)
        box = o["box"]
        bw, bh = ((box[2] - box[0]) * self.sc.s_px, (box[3] - box[1]) * self.sc.s_px) if box is not None else (np.nan, np.nan)
        L = info.get("laser_uv")
        Lxy = self.sc.to_cm(L) if L is not None else (np.nan, np.nan)
        L = L if L is not None else (np.nan, np.nan)
        ref = info.get("ref", info.get("q", (np.nan, np.nan)))
        vd = v_des if v_des is not None else (np.nan, np.nan)
        rc = self.rc_sent
        f = lambda x: "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.3f}" if isinstance(x, (float, np.floating)) else x)
        row = [self.t, self.frame_i, self.phase, int(self.seen), uv[0], uv[1], p[0], p[1], self.v_est[0], self.v_est[1],
               info.get("s"), info.get("e_n"), ref[0], ref[1], info.get("v_ff"), vd[0], vd[1], *rc, L[0], L[1], Lxy[0], Lxy[1],
               bw, bh, o["conf"], o["ms"], loop_ms, st.get("bat"), st.get("h"), st.get("tof"), st.get("vgx"), st.get("vgy"),
               st.get("vgz"), st.get("pitch"), st.get("roll"), st.get("yaw"), info.get("progress"), info.get("excite_axis"), info.get("excite_u")]
        self.logw.writerow([f(float(x)) if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else f(x) for x in row])
        if self.frame_i % 30 == 0:
            self.logf.flush()
        if info.get("e_n") is not None and self.phase == "TRACK":
            self.track_err.append(info["e_n"])

    def dash_state(self, o, info, fps):
        e = np.array(self.track_err[-2000:]) if self.track_err else np.array([np.nan])
        st = {"phase": self.phase + (" (paused)" if self.paused else ""), "t": self.t, "uv": o["uv"], "box": o["box"],
              "xy": self.p if self.p is not None else (np.nan, np.nan), "e_n": info.get("e_n", np.nan),
              "rms": float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else np.nan,
              "emax": float(np.nanmax(np.abs(e))) if np.isfinite(e).any() else np.nan,
              "progress": info.get("progress", 0.0), "v_ref": info.get("v_ff", 0.0) or 0.0, "rc": self.rc_sent,
              "bat": (getattr(self.tello, "state", {}) or {}).get("bat", np.nan), "fps": fps, "lat_ms": o["ms"], "msg": self.msg,
              "ref_xy": info.get("ref", info.get("q")), "laser_uv": info.get("laser_uv"),
              "e_xy": info.get("e_xy")}
        st.update(self.st_extra)
        return st

    def shutdown(self):
        try:
            if self.tello is not None and getattr(self.tello, "flying", False) and self.phase != "DONE":
                self.log_event("shutdown: landing")
                self.tello.land()
        except Exception:
            traceback.print_exc()
        for c in ("logf",):
            if hasattr(self, c):
                getattr(self, c).close()
        if hasattr(self, "dash"):
            self.dash.close()
        if hasattr(self, "src"):
            self.src.release()
        if self.tello is not None:
            self.tello.close()
        self.save_meta()
        try:
            from .analysis import analyze_run
            analyze_run(self.run_dir)
        except Exception as e:
            print("analysis failed:", e)
            traceback.print_exc()
        print(f"\nrun saved in {self.run_dir}")

    def save_meta(self):
        if not hasattr(self, "sc"):
            return
        sc = self.sc
        meta = {"name": self.name, "kind": self.kind, "sim": self.a.sim, "args": vars(self.a), "rig": self.rig, "model": self.model,
                "gains": self.gains, "events": self.events, "land_reason": self.land_reason, "probe": self.probe_info,
                "B_used": np.asarray(self.act.B).tolist() if hasattr(self, "act") else None,
                "scene": {"cm_per_px": sc.s_px, "up_img": sc.up.tolist(), "origin_px": sc.origin_px.tolist(), "A": sc.A.tolist(),
                          "path_cm": sc.path.pts.round(2).tolist(), "closed": sc.path.closed, "length_cm": sc.path.length,
                          "blue_cm": [sc.to_cm(b).tolist() for b in sc.blue_px],
                          "obstacle_cm": sc.to_cm(sc.obstacle_px).tolist() if sc.obstacle_px is not None else None,
                          "info": {k: v for k, v in sc.info.items()}},
                "extra": self.meta_extra()}
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
        if hasattr(self, "first_frame"):
            cv2.imwrite(str(self.run_dir / "scene_frame.jpg"), self.first_frame)

    def meta_extra(self):
        return {}
