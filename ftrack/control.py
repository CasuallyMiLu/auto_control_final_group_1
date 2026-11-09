"""Controllers. Everything is computed in wall coordinates (cm, X right, Y up) as a desired velocity (cm/s);
Actuator turns that into Tello rc values with the identified input matrix B (v_ss = B @ [rc_a, rc_c]).

With that inversion each axis seen by the PID is the normalised plant  P(s) = e^{-Ls} / (s (tau s + 1))
(desired velocity -> position), identified by sysid (MATLAB procest 'P1DI'), and the PID gains designed for it
(MATLAB pidtune, design_pid.m) have units:  Kp [1/s], Ki [1/s^2], Kd [-], N [rad/s].
"""
from __future__ import annotations

import math

import numpy as np


class PID:
    """C(s) = Kp + Ki/s + Kd N s/(s+N) (Simulink 'Parallel' form), discretised with the actual dt (backward Euler).
    Anti-windup: integration stops while the output is saturated in the direction of the error; integral only
    accumulates when |e| < i_zone (no wind-up during the approach transient)."""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, N=10.0, i_zone=np.inf, i_max=np.inf, **_):
        self.Kp, self.Ki, self.Kd, self.N = Kp, Ki, Kd, N
        self.i_zone, self.i_max = i_zone, i_max
        self.reset()

    def reset(self):
        self.I, self.D, self.e_prev = 0.0, 0.0, None

    def __call__(self, e, dt, sat=None):
        dt = max(1e-3, min(dt, 0.2))
        if self.e_prev is None:
            self.e_prev = e
        self.D = (self.D + self.Kd * self.N * (e - self.e_prev)) / (1 + self.N * dt)
        self.e_prev = e
        u = self.Kp * e + self.I + self.D
        saturated = sat is not None and abs(u) >= sat and np.sign(u) == np.sign(e)
        if abs(e) < self.i_zone and not saturated:
            self.I = float(np.clip(self.I + self.Ki * e * dt, -self.i_max, self.i_max))
        return u

    def blend(self, other: "PID", w: float) -> dict:
        return {k: (1 - w) * getattr(self, k) + w * getattr(other, k) for k in ("Kp", "Ki", "Kd", "N")}


class Actuator:
    """Desired wall velocity (cm/s) -> rc (a, b, c, d). B is 2x2 (cm/s per rc unit) mapping [a, c] to [vX, vY]."""

    def __init__(self, B, rc_max=60, accel_max=200.0):
        self.B = np.asarray(B, float)
        self.Binv = np.linalg.inv(self.B)
        self.rc_max = rc_max
        self.accel_max = accel_max
        self.v_prev = np.zeros(2)

    def vmax(self):
        """Largest wall speed reachable in every direction with |rc| <= rc_max."""
        return self.rc_max / np.abs(self.Binv).sum(1).max()

    def __call__(self, v_des, dt, b=0.0, d=0.0):
        v_des = np.asarray(v_des, float)
        dv = v_des - self.v_prev
        lim = self.accel_max * max(dt, 1e-3)
        n = np.linalg.norm(dv)
        if n > lim:
            v_des = self.v_prev + dv * lim / n
        rc = self.Binv @ v_des
        m = np.max(np.abs(rc))
        if m > self.rc_max:  # scale the vector, keep the direction
            rc *= self.rc_max / m
            v_des = self.B @ rc
        self.v_prev = v_des
        return (float(rc[0]), float(b), float(rc[1]), float(d)), v_des

    def reset(self):
        self.v_prev = np.zeros(2)


class PathFollower:
    """v_des = v_ff(s) * t  +  u_n * n
       e_n = signed cross-track error (cm, + = left of the travel direction), u_n = -PID_n(e_n)
       v_ff from the path speed profile (curvature + acceleration limits), reduced while far from the path so the
       approach from the take-off point is a clean step response of the cross-track loop."""

    def __init__(self, path, gains: dict, closed_laps: float = 1.0):
        self.path = path
        g = gains["path"]
        self.g = g
        self.pid_x = PID(**gains["cross"]["x"], i_zone=g.get("i_zone", 15), i_max=g.get("i_max", 20))
        self.pid_y = PID(**gains["cross"]["y"], i_zone=g.get("i_zone", 15), i_max=g.get("i_max", 20))
        self.pid = PID(**gains["cross"]["x"], i_zone=g.get("i_zone", 15), i_max=g.get("i_max", 20))
        path.speed_profile(g["v_max"], g["a_lat"], g["a_long"], g.get("v_min", 6.0))
        self.s = None
        self.travel = 0.0
        self.target = closed_laps * path.length if path.closed else path.length
        self.done = False
        self.reached = False  # first time |e_n| < reach_cm

    def reset(self):
        self.s, self.travel, self.done = None, 0.0, False
        self.pid.reset()

    def step(self, p, dt):
        pr = self.path.project(p, self.s, back=25.0, fwd=self.g.get("search_fwd", 40.0))
        if self.s is not None:
            ds = pr["s"] - self.s
            if self.path.closed:
                ds = (ds + self.path.length / 2) % self.path.length - self.path.length / 2
            self.travel += max(ds, -5.0)  # progress (backwards motion counts only a little)
        self.s = pr["s"]
        e = pr["e_n"]
        n = pr["nrm"]
        # gains for this normal direction: blend the X and Y designs by the direction of the correction
        w = float(n[1] ** 2)
        for k, v in self.pid_x.blend(self.pid_y, w).items():
            setattr(self.pid, k, v)
        u_n = -self.pid(e, dt)
        e0 = self.g.get("e_ff0", 30.0)
        approach = float(np.clip(1.0 - abs(e) / e0, 0.0, 1.0))
        if abs(e) < self.g.get("reach_cm", 5.0):
            self.reached = True
        v_ff = self.path.speed_at(self.s + self.g.get("ff_lead_cm", 5.0)) * approach
        if not self.path.closed:  # slow down to stop at the end of an open path
            rem = self.path.length - self.s
            v_ff = min(v_ff, max(0.0, math.sqrt(2 * self.g["a_long"] * max(rem - 2.0, 0.0))))
        t_hat = self.path.tangent(self.s + self.g.get("ff_lead_cm", 5.0))
        v = v_ff * t_hat + u_n * n
        if self.travel >= self.target - (0 if self.path.closed else 3.0) or (not self.path.closed and self.path.length - self.s < 3.0):
            self.done = True
        return v, {"s": self.s, "e_n": e, "q": pr["q"], "tan": pr["tan"], "nrm": n, "v_ff": v_ff,
                   "progress": min(1.0, self.travel / max(self.target, 1e-6)), "u_n": u_n}


class PointTracker:
    """Per-axis PID towards a target point (laser dot, hover point). Target is low-pass filtered (hand jitter)."""

    def __init__(self, gains: dict, key="point"):
        g = gains[key]
        self.g = g
        self.pid = [PID(**gains["point_axes"]["x"], i_zone=g.get("i_zone", 12), i_max=g.get("i_max", 15)),
                    PID(**gains["point_axes"]["y"], i_zone=g.get("i_zone", 12), i_max=g.get("i_max", 15))]
        self.tf = None
        self.v_max = g.get("v_max", 40.0)

    def reset(self):
        for p in self.pid:
            p.reset()
        self.tf = None

    def step(self, p, target, dt):
        target = np.asarray(target, float)
        tau = self.g.get("target_tau", 0.15)
        self.tf = target if self.tf is None else self.tf + (target - self.tf) * min(1.0, dt / max(tau, 1e-3))
        e = self.tf - np.asarray(p, float)
        if np.linalg.norm(e) < self.g.get("deadband", 0.0):
            e = np.zeros(2)
        v = np.array([self.pid[0](e[0], dt, self.v_max), self.pid[1](e[1], dt, self.v_max)])
        n = np.linalg.norm(v)
        if n > self.v_max:
            v *= self.v_max / n
        return v, {"ref": self.tf.copy(), "e": e}


# ------------------------------------------------------------------------------------------- design fallback
def design_pid_numeric(tau, L, wc, pm_deg, ki_ratio=0.15, N=None):
    """PID for P(s) = e^{-Ls}/(s(tau s+1)) with crossover wc and phase margin pm (Python fallback for pidtune).
    PIDF: C = Kp (1 + wi/s) + Kd N s/(s+N). Choose wi = ki_ratio*wc, solve Kp, Kd for |CP|=1, angle = -180+pm."""
    N = N or 10 * wc
    s = 1j * wc
    P = np.exp(-L * s) / (s * (tau * s + 1))
    target = -1.0 / P * np.exp(1j * math.radians(pm_deg))  # C(jwc) needed: C P = -e^{j pm}
    wi = ki_ratio * wc
    # C = Kp (1 + wi/s) + Kd * N s/(s+N): linear in Kp, Kd -> solve 2 real equations
    a = 1 + wi / s
    b = N * s / (s + N)
    M = np.array([[a.real, b.real], [a.imag, b.imag]])
    Kp, Kd = np.linalg.solve(M, [target.real, target.imag])
    return {"Kp": float(Kp), "Ki": float(Kp * wi), "Kd": float(max(Kd, 0.0)), "N": float(N)}
