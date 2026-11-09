"""主題 1：循跡飛行 (circle / square). One program: start -> take-off -> approach the white centre line -> one lap
-> land.  python theme1.py --scene circle      (add --sim for a dry run without the drone)"""
from __future__ import annotations

import numpy as np

import numpy as np

from ftrack.control import PathFollower, PointTracker
from ftrack.mission import Mission, base_args


class Theme1(Mission):
    name = "theme1"

    def __init__(self, args):
        self.kind = args.scene
        self.name = f"theme1_{args.scene}"
        super().__init__(args)

    def title(self):
        return f"Theme 1 - {self.kind} tracking"

    def build(self):
        laps = self.rig.get("closed_path", {}).get("laps", 1.0)
        self.fol = PathFollower(self.sc.path, self.gains, closed_laps=laps)
        self.pt = PointTracker(self.gains, "point")
        # staging point: a fixed distance outside the start of the track (radially from the track centre), so the
        # release into tracking is the same step input e0 every run (rise / overshoot / settling are comparable)
        P = self.sc.path.pts
        c = 0.5 * (P.min(0) + P.max(0))
        d = P[0] - c
        self.stage_xy = P[0] + self.a.stage_cm * d / (np.linalg.norm(d) + 1e-9)

    def first_phase(self):
        return "STAGE" if self.a.stage_cm > 0 else "TRACK"

    def on_phase(self, ph):
        if ph == "STAGE":
            self.pt.reset()
            self.act.reset()
            self.near_since = None
        if ph == "TRACK":
            self.fol.reset()
            self.act.reset()
            self.track_t0 = self.t

    def control(self, p, dt):
        if self.phase == "STAGE":
            v, info = self.pt.step(p, self.stage_xy, dt)
            info["ref"] = self.stage_xy
            info["e_xy"] = tuple(self.stage_xy - p)
            ok = np.linalg.norm(p - self.stage_xy) < 4 and np.linalg.norm(self.v_est) < 8
            self.near_since = (self.near_since or self.t) if ok else None
            if (self.near_since is not None and self.t - self.near_since > 0.8) or self.t - self.phase_t0 > 12:
                self.set_phase("TRACK")
            return v, info
        v, info = self.fol.step(p, dt)
        info["e_xy"] = tuple(info["q"] - p)
        if self.fol.done:
            self.log_event(f"lap complete in {self.t - self.track_t0:.1f} s")
            self.land("lap complete")
        return v, info

    def sim_start_xy(self, sc):
        # course sheet: take-off point bottom-right, below the track, on the floor (~135 cm below the centre).
        # Keep it inside the camera's horizontal view (zoomed + rotated camera: only ~+-105 cm sideways)
        lo = sc.path.pts.min(0)
        hi = sc.path.pts.max(0)
        return (0.8 * hi[0], lo[1] - 60.0)

    def meta_extra(self):
        return {"track_t0": getattr(self, "track_t0", None)}


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--scene", choices=["circle", "square"], required=True)
    ap.add_argument("--stage-cm", type=float, default=25.0, help="hold this far outside the start before tracking (step size); 0 = off")
    Theme1(ap.parse_args()).run()
