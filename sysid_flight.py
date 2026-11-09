"""系統辨識飛行 (Lab1-style identification on the real drone, camera = sensor).

take-off -> hold at the centre of the board (closed loop) -> for each wall axis (X = rc a, Y = rc c):
open-loop doublets (+A for T, rest, -A for T, rest) at several amplitudes/durations and a short random
binary (PRBS-like) sequence, re-centring in closed loop between blocks -> land.
Output runs/<run>/log.csv (t, rc, X, Y, ...). Then: matlab/sysid_tello.m (procest P1DI + tfest) -> config/model.json,
matlab/design_pid.m (pidtune) -> config/gains.json.      python sysid_flight.py --scene circle   (--sim to test)
"""
from __future__ import annotations

import numpy as np

from ftrack.control import PointTracker
from ftrack.mission import Mission, base_args


def make_program(rng, amps=(25, 40), durs=(0.8, 1.3), rest=1.2, prbs_s=7.0, prbs_amp=30, prbs_dt=0.35):
    """List of blocks; each block is a list of (duration, u). Doublets start with alternating sign."""
    blocks = []
    sgn = 1
    for A in amps:
        for T in durs:
            blocks.append([(T, sgn * A), (rest, 0), (T, -sgn * A), (rest, 0)])
            sgn = -sgn
    seq = []
    t = 0.0
    u = prbs_amp
    while t < prbs_s:
        d = prbs_dt * rng.integers(1, 4)
        seq.append((d, u))
        u = -u if rng.random() < 0.7 else u
        t += d
    seq.append((rest, 0))
    blocks.append(seq)
    return blocks


class SysID(Mission):
    name = "sysid"

    def __init__(self, args):
        self.kind = args.scene
        self.name = f"sysid_{args.scene}"
        super().__init__(args)
        rng = np.random.default_rng(args.seed)
        self.axes = [a for a in args.axes.split(",") if a]
        self.prog = {ax: make_program(rng, amps=tuple(args.amps), durs=tuple(args.durs)) for ax in self.axes}
        self.queue = [(ax, i) for ax in self.axes for i in range(len(self.prog[ax]))]
        self.box = args.box_cm

    def title(self):
        return "System identification flight"

    def build(self):
        self.pt = PointTracker(self.gains, "point")
        P = self.sc.path.pts
        # hold point: centre of the track, shifted off the black centre cross (a dark drone on a dark cross is hard
        # to see for background subtraction)
        self.centre = 0.5 * (P.min(0) + P.max(0)) + np.array(self.a.hold_offset, float)

    def first_phase(self):
        return "GOTO"

    def on_phase(self, ph):
        if ph == "GOTO":
            self.pt.reset()
            self.act.reset()
            self.near_since = None
        if ph == "EXCITE":
            self.blk_t0 = self.t

    def control(self, p, dt):
        if self.phase == "GOTO":
            v, info = self.pt.step(p, self.centre, dt)
            info["ref"] = self.centre
            info["e_xy"] = tuple(self.centre - p)
            if np.linalg.norm(p - self.centre) < 6 and np.linalg.norm(self.v_est) < 8:
                self.near_since = self.t if self.near_since is None else self.near_since
                if self.t - self.near_since > 1.0:
                    if not self.queue:
                        self.land("identification program finished")
                    else:
                        self.cur = self.queue.pop(0)
                        self.set_phase("EXCITE")
            else:
                self.near_since = None
            if self.t - self.phase_t0 > 15:
                self.log_event("could not settle at the centre in 15 s; skipping the rest")
                self.land("GOTO timeout")
            return v, info
        if self.phase == "EXCITE":
            ax, i = self.cur
            blk = self.prog[ax][i]
            tb = self.t - self.blk_t0
            acc, u = 0.0, None
            for d, uu in blk:
                acc += d
                if tb < acc:
                    u = uu
                    break
            dev = p - self.centre
            if u is None or np.max(np.abs(dev)) > self.box:
                if u is not None:
                    self.log_event(f"block {ax}{i} stopped: {np.max(np.abs(dev)):.0f} cm from the centre")
                self.set_phase("GOTO")
                return None, {"rc": (0, 0, 0, 0)}
            rc = (u, 0, 0, 0) if ax == "x" else (0, 0, u, 0)
            return None, {"rc": rc, "excite_axis": ax, "excite_u": u, "ref": self.centre}
        return np.zeros(2), {}

    def sim_start_xy(self, sc):
        P = sc.path.pts
        return (P.max(0)[0] + 25.0, P.min(0)[1] - 60.0)


if __name__ == "__main__":
    ap = base_args(__doc__)
    ap.add_argument("--scene", choices=["circle", "square", "path"], default="circle")
    ap.add_argument("--axes", default="x,y")
    ap.add_argument("--amps", type=float, nargs="+", default=[25, 40])
    ap.add_argument("--durs", type=float, nargs="+", default=[0.8, 1.3])
    ap.add_argument("--box-cm", type=float, default=65.0, help="abort a block when the drone is this far from the centre")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-offset", type=float, nargs=2, default=[0.0, 30.0], help="hold point relative to the track centre (cm)")
    SysID(ap.parse_args()).run()
