"""Reference path in wall coordinates (cm, X right, Y up): projection, cross-track error, speed profile."""
from __future__ import annotations

import numpy as np


class Path:
    def __init__(self, pts: np.ndarray, closed: bool, ds: float = 1.0):
        pts = np.asarray(pts, float)
        self.closed = closed
        self.pts = resample(pts, ds, closed)
        seg = np.diff(np.vstack([self.pts, self.pts[:1]]) if closed else self.pts, axis=0)
        self.seglen = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0], np.cumsum(self.seglen)])  # arclength at each vertex (+ closing vertex)
        self.length = float(self.s[-1])
        self.seg = seg
        self.tan = seg / np.maximum(self.seglen[:, None], 1e-9)
        self.kappa = self._curvature()
        self.vmax = None

    # ------------------------------------------------------------------ geometry
    def _curvature(self, win_cm: float = 6.0) -> np.ndarray:
        """Curvature (1/cm) per vertex from the heading change over +-win_cm."""
        n = len(self.pts)
        k = max(1, int(round(win_cm / max(np.median(self.seglen), 1e-6))))
        raw = np.arctan2(self.tan[:, 1], self.tan[:, 0])
        if self.closed:
            hp = np.unwrap(np.concatenate([raw[-k:], raw, raw[:k]]))
        else:
            head = np.unwrap(raw)
            hp = np.concatenate([np.full(k, head[0]), head, np.full(k, head[-1])])
        dh = hp[2 * k:] - hp[:-2 * k]
        kap = np.abs(dh) / (2 * k * np.median(self.seglen))
        if len(kap) < n:
            kap = np.concatenate([kap, np.full(n - len(kap), kap[-1])])
        return kap[:n]

    def point(self, s: float) -> np.ndarray:
        s = self.wrap(s)
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seglen) - 1))
        return self.pts[i] + self.tan[i] * (s - self.s[i])

    def tangent(self, s: float) -> np.ndarray:
        s = self.wrap(s)
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seglen) - 1))
        return self.tan[i]

    def wrap(self, s: float) -> float:
        return s % self.length if self.closed else float(np.clip(s, 0, self.length))

    def project(self, p, s_hint: float | None = None, back: float = 25.0, fwd: float = 60.0):
        """Closest point to p. With s_hint, only segments within [s_hint-back, s_hint+fwd] are searched, so the
        progress variable never jumps to another part of the path (corners of the square, the 'S' of theme 2)."""
        p = np.asarray(p, float)
        a = self.pts[: len(self.seglen)]
        t = np.einsum("ij,ij->i", p - a, self.seg) / np.maximum(self.seglen ** 2, 1e-12)
        t = np.clip(t, 0, 1)
        q = a + self.seg * t[:, None]
        d2 = np.sum((p - q) ** 2, axis=1)
        if s_hint is not None:
            sv = self.s[:-1] + t * self.seglen
            ds = sv - s_hint
            if self.closed:
                ds = (ds + self.length / 2) % self.length - self.length / 2
            d2 = np.where((ds >= -back) & (ds <= fwd), d2, np.inf)
            if not np.isfinite(d2).any():
                return self.project(p)
        i = int(np.argmin(d2))
        s = float(self.s[i] + t[i] * self.seglen[i])
        tan = self.tan[i]
        nrm = np.array([-tan[1], tan[0]])  # left normal
        e_n = float(np.dot(p - q[i], nrm))  # signed cross-track error (+ = left of travel direction)
        return {"s": s, "q": q[i], "tan": tan, "nrm": nrm, "e_n": e_n, "dist": float(np.sqrt(d2[i])), "kappa": float(self.kappa[i])}

    def distance(self, P: np.ndarray) -> np.ndarray:
        """Unsigned distance of many points to the path (for analysis)."""
        P = np.atleast_2d(P)
        a = self.pts[: len(self.seglen)]
        out = np.empty(len(P))
        for j, p in enumerate(P):
            t = np.clip(np.einsum("ij,ij->i", p - a, self.seg) / np.maximum(self.seglen ** 2, 1e-12), 0, 1)
            out[j] = np.sqrt(np.min(np.sum((p - (a + self.seg * t[:, None])) ** 2, axis=1)))
        return out

    # ------------------------------------------------------------------ speed profile
    def speed_profile(self, v_max: float, a_lat: float, a_long: float, v_min: float = 5.0) -> np.ndarray:
        """v(s) <= sqrt(a_lat / kappa), then forward/backward passes limit dv/ds by a_long (v dv/ds = a)."""
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(self.kappa, 1e-6)))
        v = np.maximum(v, v_min)
        ds = np.concatenate([self.seglen, self.seglen[-1:]])[: len(v)]
        for i in range(1, len(v)):  # accelerate
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_long * ds[i - 1]))
        for i in range(len(v) - 2, -1, -1):  # brake
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long * ds[i]))
        if not self.closed:
            v[-1] = v_min
        self.vmax = v
        return v

    def speed_at(self, s: float) -> float:
        i = int(np.clip(np.searchsorted(self.s, self.wrap(s), side="right") - 1, 0, len(self.vmax) - 1))
        return float(self.vmax[i])


def resample(pts: np.ndarray, ds: float, closed: bool) -> np.ndarray:
    P = np.vstack([pts, pts[:1]]) if closed else pts
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-9])
    P = P[keep]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(int(np.ceil(s[-1] / ds)), 4)
    si = np.linspace(0, s[-1], n + 1)
    out = np.column_stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])])
    return out[:-1] if closed else out


def smooth_polyline(pts: np.ndarray, win: int, closed: bool) -> np.ndarray:
    if win < 2:
        return pts
    k = np.ones(win) / win
    if closed:
        P = np.vstack([pts[-win:], pts, pts[:win]])
        return np.column_stack([np.convolve(P[:, i], k, "same") for i in range(2)])[win:-win]
    P = np.vstack([np.repeat(pts[:1], win, 0), pts, np.repeat(pts[-1:], win, 0)])
    out = np.column_stack([np.convolve(P[:, i], k, "same") for i in range(2)])[win:-win]
    out[0], out[-1] = pts[0], pts[-1]
    return out
