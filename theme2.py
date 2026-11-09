"""主題 2：循跡導引計時賽. One program: start -> take-off -> follow the red path A -> B (theme-1 controller)
-> follow the green laser dot (operator steers around the obstacle) -> when the drone holds over the lower blue
square (C, above the platform D) -> land.     python theme2.py        (add --sim for a dry run)"""
from __future__ import annotations

import numpy as np

from ftrack.control import PathFollower, PointTracker
from ftrack.mission import Mission, base_args
from ftrack.vision import LaserDetector


class Theme2(Mission):
    name = "theme2"
    kind = "path"

    def title(self):
        return "Theme 2 - path + laser race"

    def build(self):
        self.fol = PathFollower(self.sc.path, self.gains)
        self.pt = PointTracker(self.gains, "point")
        lo, hi = self.sc.bounds_cm(self.rig["vision"].get("work_margin_cm", 60))
        poly = self.sc.to_px(np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]))
        self.laser = LaserDetector(poly, self.sc.s_px)
        cfg = self.rig["theme2"]
        self.land_xy = self.landing_point(self.sc) + np.array(cfg.get("land_offset_cm", [0, 0]), float)
        self.land_r = cfg.get("land_radius_cm", 15)
        self.st_extra = {"land_zone_xy": self.land_xy, "land_r": self.land_r}
        self.in_zone_since = None
        self.laser_last = None
        self.laser_last_t = -1e9
        self.hold_xy = None

    @staticmethod
    def landing_point(sc):
        """C/D: the blue square that is NOT next to the end B of the track (the lower one)."""
        B = sc.path.pts[-1]
        blues = [sc.to_cm(b) for b in sc.blue_px]
        if not blues:
            print("[theme2] WARNING: no blue squares found; landing point = 60 cm left of the start A")
            return sc.path.pts[0] + np.array([-60.0, 0.0])
        return max(blues, key=lambda b: np.linalg.norm(b - B))

    def first_phase(self):
        return "TRACK"

    def on_phase(self, ph):
        if ph == "TRACK":
            self.fol.reset()
            self.act.reset()
            self.track_t0 = self.t
        if ph == "LASER":
            self.pt.reset()
            self.hold_xy = self.p.copy() if self.p is not None else self.sc.path.pts[-1].copy()
            self.laser_t0 = self.t

    def control(self, p, dt):
        if self.phase == "TRACK":
            v, info = self.fol.step(p, dt)
            info["e_xy"] = tuple(info["q"] - p)
            if self.fol.done:
                self.log_event(f"reached B (path done) after {self.t - self.track_t0:.1f} s")
                self.set_phase("LASER")
            return v, info
        if self.phase == "LASER":
            L = self.laser(self.frame_now)
            info = {}
            if L is not None:
                self.laser_last, self.laser_last_t = self.sc.to_cm(L), self.t
                info["laser_uv"] = L
            if self.t - self.laser_last_t <= self.rig["theme2"].get("laser_lost_hold_s", 0.6) and self.laser_last is not None:
                target = self.laser_last
                self.hold_xy = None
            else:  # no dot: hold where we are
                if self.hold_xy is None:
                    self.hold_xy = p.copy()
                target = self.hold_xy
            v, ti = self.pt.step(p, target, dt)
            info["ref"] = ti["ref"]
            info["e_xy"] = tuple(ti["e"])
            # landing trigger (rig theme2.land_trigger): zone = over the lower blue square; laser_off = operator
            # switches the laser off after using it; key = L key only. The obstacle is never used.
            trig = self.rig["theme2"].get("land_trigger", "zone")
            if trig == "laser_off":
                if self.laser_last is not None and self.t - self.laser_last_t > self.rig["theme2"].get("laser_off_s", 0.7):
                    self.log_event(f"laser switched off -> land (laser phase {self.t - self.laser_t0:.1f} s)")
                    self.land("laser off")
            elif trig == "zone" and np.linalg.norm(p - self.land_xy) < self.land_r:
                self.in_zone_since = self.t if self.in_zone_since is None else self.in_zone_since
                self.msg = f"over the landing zone {self.t - self.in_zone_since:.1f} s"
                if self.t - self.in_zone_since >= self.rig["theme2"].get("land_hold_s", 0.6):
                    self.log_event(f"landing zone reached (laser phase {self.t - self.laser_t0:.1f} s)")
                    self.land("landing zone")
            elif trig == "zone":
                self.in_zone_since = None
            return v, info
        return np.zeros(2), {}

    def loop(self):
        # keep the latest frame for the laser detector (vision is called in the base loop)
        orig = self.vision

        def vis(frame, t):
            self.frame_now = frame
            return orig(frame, t)

        self.vision = vis
        try:
            super().loop()
        finally:
            self.vision = orig

    # ----------------------------------------------------------------------------- simulation helpers
    def sim_start_xy(self, sc):
        A = sc.path.pts[0]
        return (A[0], A[1] - 25.0 - 55.0)  # on the floor below A (Tello climbs ~80 cm on take-off)

    def sim_laser(self):
        """Simulated operator: once the drone reaches B, the dot leads the drone ~25 cm towards the next waypoint
        (left of B, down the left side of the obstacle, over the landing square)."""
        sc = None
        state = {"on": False, "k": 0, "wp": None}

        def laser(t):
            nonlocal sc
            sc = self.sc_true
            d = self.sim_drone.p
            B = sc.path.pts[-1]
            if state["wp"] is None:
                land = self.landing_point(sc)
                if sc.obstacle_px is not None:
                    O = sc.to_cm(sc.obstacle_px)
                    xmin, ymin, ymax = O[:, 0].min(), O[:, 1].min(), O[:, 1].max()
                else:
                    xmin, ymin, ymax = B[0] - 60, B[1] - 60, B[1] - 30
                state["wp"] = [np.array([xmin - 30, max(B[1], ymax + 15)]), np.array([xmin - 30, ymin - 25]), np.array(land)]
            if not state["on"]:
                if np.linalg.norm(d - B) < 25:
                    state["on"] = True
                else:
                    return None
            wp = state["wp"]
            if state["k"] < len(wp) - 1 and np.linalg.norm(d - wp[state["k"]]) < 12:
                state["k"] += 1
            w = wp[state["k"]]
            v = w - d
            n = np.linalg.norm(v)
            return w if n < 25 else d + v / n * 25

        return laser

    def meta_extra(self):
        return {"land_xy": self.land_xy.tolist() if hasattr(self, "land_xy") else None}


if __name__ == "__main__":
    Theme2(base_args(__doc__).parse_args()).run()
