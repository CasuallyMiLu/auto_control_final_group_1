"""Grey-box identification of the Tello wall-plane dynamics from a sysid_flight.py log (Python twin of
matlab/sysid_tello.m, so results can be checked without MATLAB).

Model per axis (process model 'P1DI' in MATLAB terms):   X(s)/U(s) = Kv e^{-L s} / (s (tau s + 1))
  U = rc value (-100..100), X = wall position (cm).  Kv = steady speed per rc unit (cm/s), tau = speed lag, L = dead
  time (Tello + Wi-Fi + camera + vision).
Data: EXCITE phases only (open loop), each with 0.6 s of preceding commands (still in the delay pipe) and 1 s of
the following samples. Each block is simulated from its own initial position/velocity plus a constant drift
velocity (|w| <= 8 cm/s; hover drift / air flow), which are nuisance parameters; Kv, tau, L are shared.
Training: the PRBS block + the smaller-amplitude doublets.  Validation: the larger-amplitude doublets (never used
in the fit).  Fit% = 100 (1 - |y - y_hat| / |y - mean(y)|)  (MATLAB compare() definition).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DRIFT = 8.0  # cm/s: bound of the constant per-block drift (air flow / hover drift), a nuisance parameter like detrend


def blocks_from_log(df, axis):
    ex = df[(df["phase"] == "EXCITE") & (df["excite_axis"] == axis)].copy()
    if ex.empty:
        return []
    # a new block starts when the time gap > 0.3 s (GOTO in between)
    t = ex["t"].to_numpy(float)
    cut = np.nonzero(np.diff(t) > 0.3)[0] + 1
    out = []
    for idx in np.split(np.arange(len(ex)), cut):
        b = ex.iloc[idx]
        # include 1.5 s of the following samples (the drone keeps moving after the last command)
        t_end, t_beg = b["t"].iloc[-1], b["t"].iloc[0]
        tail = df[(df["t"] > t_end) & (df["t"] <= t_end + 1.0)]
        pre = df[(df["t"] < t_beg) & (df["t"] >= t_beg - 0.6)]  # commands still in the delay pipe at the start
        b = pd.concat([pre, b, tail])
        n_pre = len(pre)
        y = b["X" if axis == "x" else "Y"].to_numpy(float)
        u = b["rc_a" if axis == "x" else "rc_c"].to_numpy(float)  # logged commands, incl. the re-centring tail
        tt = b["t"].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        ue = u[n_pre:n_pre + len(idx)]
        amp = float(np.max(np.abs(ue)))
        is_prbs = np.sum(np.abs(np.diff(np.sign(ue))) > 0) > 6
        out.append({"t": tt, "u": u, "y": y, "amp": amp, "prbs": bool(is_prbs)})
    return out


def simulate(t, u, K, tau, L, x0, v0, w=0.0):
    """Exact zero-order-hold integration of tau v' = K u(t-L) - v, x' = v on the (irregular) time grid."""
    x = np.empty(len(t))
    v = v0
    xc = x0
    x[0] = xc
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        # input active during [t_{i-1}, t_i): u at time t_{i-1} - L (last command sent before that moment)
        j = np.searchsorted(t, t[i - 1] - L, side="right") - 1
        ui = u[j] if j >= 0 else 0.0
        vs = K * ui
        a = np.exp(-h / tau)
        xc = xc + vs * h + (v - vs) * tau * (1 - a)
        v = vs + (v - vs) * a
        x[i] = xc
    return x + w * (t - t[0])


def fit_axis(blocks, train_idx, L_grid=None):
    def resid(th, idx):
        K, tau, L = th[:3]
        r = []
        for k, i in enumerate(idx):
            b = blocks[i]
            x0, v0, w = th[3 + 3 * k], th[4 + 3 * k], th[5 + 3 * k]
            yh = simulate(b["t"], b["u"], K, tau, L, x0, v0, w)
            ok = np.isfinite(b["y"])
            r.append(yh[ok] - b["y"][ok])
        return np.concatenate(r)

    def init_state(b):
        ok = np.isfinite(b["y"])
        y, t = b["y"][ok], b["t"][ok]
        v0 = (y[min(3, len(y) - 1)] - y[0]) / max(t[min(3, len(t) - 1)] - t[0], 1e-3)
        return [y[0], float(np.clip(v0, -150, 150)), 0.0]

    best = None
    for L0 in (0.1, 0.2, 0.3):
        th0 = [1.0, 0.35, L0] + sum((init_state(blocks[i]) for i in train_idx), [])
        lo = [0.05, 0.03, 0.0] + [-np.inf, -200, -DRIFT] * len(train_idx)
        hi = [5.0, 3.0, 0.8] + [np.inf, 200, DRIFT] * len(train_idx)
        r = least_squares(resid, th0, bounds=(lo, hi), args=(train_idx,), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    K, tau, L = best.x[:3]
    return {"Kv": float(K), "tau": float(tau), "L": float(L)}


def fit_pct(y, yh):
    ok = np.isfinite(y)
    y, yh = y[ok], yh[ok]
    return float(100 * (1 - np.linalg.norm(y - yh) / max(np.linalg.norm(y - y.mean()), 1e-9)))


def evaluate(blocks, idx, p):
    """Validation: simulate each block from its measured initial position, initial velocity fitted (2 numbers per
    block - the model parameters stay fixed)."""
    out = []
    for i in idx:
        b = blocks[i]
        ok = np.isfinite(b["y"])

        def r(s):
            return simulate(b["t"], b["u"], p["Kv"], p["tau"], p["L"], s[0], s[1], s[2])[ok] - b["y"][ok]

        y0 = b["y"][ok][0]
        s = least_squares(r, [y0, 0.0, 0.0], bounds=([-np.inf, -200, -DRIFT], [np.inf, 200, DRIFT])).x
        yh = simulate(b["t"], b["u"], p["Kv"], p["tau"], p["L"], s[0], s[1], s[2])
        out.append({"block": int(i), "amp": b["amp"], "prbs": b["prbs"], "fit_pct": fit_pct(b["y"], yh), "yhat": yh})
    return out


def identify_run(run, plot=True):
    run = Path(run)
    df = pd.read_csv(run / "log.csv")
    res = {"run": run.name, "axes": {}}
    figs = []
    for ax in ("x", "y"):
        bl = blocks_from_log(df, ax)
        if not bl:
            continue
        amps = sorted({b["amp"] for b in bl if not b["prbs"]})
        big = amps[-1] if len(amps) > 1 else None
        train = [i for i, b in enumerate(bl) if b["prbs"] or b["amp"] != big]
        valid = [i for i in range(len(bl)) if i not in train]
        p = fit_axis(bl, train)
        ev_t = evaluate(bl, train, p)
        ev_v = evaluate(bl, valid, p)
        res["axes"][ax] = {**p, "train_fit_pct": [round(e["fit_pct"], 1) for e in ev_t], "valid_fit_pct": [round(e["fit_pct"], 1) for e in ev_v],
                           "n_train_blocks": len(train), "n_valid_blocks": len(valid)}
        figs.append((ax, bl, ev_t + ev_v, set(train)))
    # B matrix (diagonal; sign from the fit is always + because X/Y are defined by the probe-corrected frame)
    if "x" in res["axes"] and "y" in res["axes"]:
        res["B"] = [[res["axes"]["x"]["Kv"], 0.0], [0.0, res["axes"]["y"]["Kv"]]]
    if plot and figs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(figs), 1, figsize=(12, 3.6 * len(figs)))
        axs = np.atleast_1d(axs)
        for a, (ax, bl, ev, train) in zip(axs, figs):
            t_off = 0.0
            for e in sorted(ev, key=lambda e: e["block"]):
                b = bl[e["block"]]
                tt = b["t"] - b["t"][0] + t_off
                col = "tab:blue" if e["block"] in train else "tab:green"
                a.plot(tt, b["y"] - b["y"][np.isfinite(b["y"])][0], color="k", lw=1)
                a.plot(tt, e["yhat"] - b["y"][np.isfinite(b["y"])][0], color=col, lw=1.5, ls="--")
                a.text(tt[0], a.get_ylim()[1] if a.lines else 0, f"{'T' if e['block'] in train else 'V'} {e['fit_pct']:.0f}%", fontsize=7, va="top")
                t_off = tt[-1] + 0.5
            p = res["axes"][ax]
            a.set_title(f"axis {ax.upper()}: Kv={p['Kv']:.3f} cm/s/rc, tau={p['tau']:.3f} s, L={p['L']:.3f} s   "
                        f"(black: measured, blue: training fit, green: validation)")
            a.set_ylabel("cm (from block start)")
        axs[-1].set_xlabel("concatenated block time (s)")
        fig.tight_layout()
        fig.savefig(run / "fig_sysid_fit.png", dpi=140)
        plt.close(fig)
    (run / "model_fit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def to_model_json(res, src):
    ax = res["axes"]
    return {"_doc": "identified by ftrack/sysid.py (python twin of matlab/sysid_tello.m)", "source": f"sysid:{src}",
            "B": res["B"], "axes": {k: {"Kv": v["Kv"], "tau": v["tau"], "L": v["L"]} for k, v in ax.items()}}
