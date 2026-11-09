"""Post-flight analysis of runs/<run>/log.csv + meta.json -> metrics.json, metrics.md, fig_*.png.

Time-domain specifications (computed by code, never read off a plot), on the cross-track error e_n(t) of the
TRACK phase (cm, + = left of the travel direction). The approach from the take-off point to the line is a step
of size e0 = e_n(t_start):
  rise time  t_r : |e_n| goes from 90 % to 10 % of |e0|        (= y(t) = 1 - e_n/e0 from 10 % to 90 %)
  overshoot  M_p : largest excursion to the other side of the line, in % of |e0| (and in cm)
  peak time  t_p : time of that excursion
  settling   t_s : first time after which |e_n| stays inside +-band, band = 5 cm (quarter of the 20 cm red
                   line); also for 3 cm. Square: the step window ends 20 cm before the first corner (each corner
                   is a new disturbance, reported separately as the max error within +-25 cm of the corner)
  steady state (t > t_s): mean e_n (= steady-state error, signed bias), RMS, max |e_n|, P95, % of time on the
                   red line (|e_n| <= 10 cm)
Distances are on the wall plane (where the path is painted); 'drone plane' values scale them by (L-d)/L, with d
the drone-to-wall distance estimated from its apparent size (pin-hole model, camera at L from the wall).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BAND = 5.0


def _load(run):
    run = Path(run)
    df = pd.read_csv(run / "log.csv")
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    return run, df, meta


def step_metrics(t, e, band=BAND):
    t = np.asarray(t, float)
    e = np.asarray(e, float)
    ok = np.isfinite(e)
    t, e = t[ok], e[ok]
    out = {"e0_cm": float(e[0]) if len(e) else np.nan}
    if len(e) < 5:
        return out
    t = t - t[0]
    e0 = e[0]
    out["band_cm"] = band
    if abs(e0) >= 2 * band:
        y = 1 - e / e0
        i10 = np.argmax(y >= 0.1) if (y >= 0.1).any() else None
        i90 = np.argmax(y >= 0.9) if (y >= 0.9).any() else None
        out["rise_time_s"] = float(t[i90] - t[i10]) if i10 is not None and i90 is not None else np.nan
        cross = np.nonzero(np.sign(e) != np.sign(e0))[0]
        if len(cross):
            seg = slice(cross[0], len(e))
            # overshoot only within the first transient (until 6 s after the first crossing)
            w = (t >= t[cross[0]]) & (t <= t[cross[0]] + 6.0)
            k = np.argmax(-np.sign(e0) * e * w)
            out["overshoot_cm"] = float(-np.sign(e0) * e[k])
            out["overshoot_pct"] = float(100 * out["overshoot_cm"] / abs(e0))
            out["peak_time_s"] = float(t[k])
        else:
            out["overshoot_cm"], out["overshoot_pct"], out["peak_time_s"] = 0.0, 0.0, np.nan
    for b, key in ((band, "settling_time_s"), (3.0, "settling_time_3cm_s")):
        outside = np.nonzero(np.abs(e) > b)[0]
        if len(outside) == 0:
            out[key] = 0.0
        elif outside[-1] == len(e) - 1:
            out[key] = np.nan  # never settled
        else:
            out[key] = float(t[outside[-1] + 1])
    ts = out.get("settling_time_s")
    if ts is None or not np.isfinite(ts):
        ts = min(t[-1] * 0.3, 8.0)  # no settling: use the last 70 % as 'steady'
        out["steady_from_s"] = float(ts)
    else:
        out["steady_from_s"] = float(ts)
    ss = e[t >= out["steady_from_s"]]
    if len(ss) > 3:
        out.update(ss_mean_cm=float(np.mean(ss)), ss_mean_abs_cm=float(np.mean(np.abs(ss))), ss_rms_cm=float(np.sqrt(np.mean(ss ** 2))),
                   ss_max_cm=float(np.max(np.abs(ss))), ss_p95_cm=float(np.percentile(np.abs(ss), 95)),
                   on_line_pct=float(100 * np.mean(np.abs(ss) <= 10.0)), on_white_pct=float(100 * np.mean(np.abs(ss) <= 2.0)))
    return out


def depth_estimate(df, meta):
    L = meta["rig"].get("camera_distance_cm", 450)
    size = meta["rig"].get("drone_size_cm", 20)
    bw = df[["box_w", "box_h"]].max(axis=1).to_numpy(float)
    bw = bw[np.isfinite(bw)]
    if len(bw) < 10:
        return None
    r = np.median(bw) / size  # apparent / real  = L / (L - d)
    d = L * (1 - 1 / r) if r > 1 else 0.0
    return {"apparent_size_cm": float(np.median(bw)), "drone_wall_dist_cm": float(d), "scale_to_drone_plane": float((L - d) / L)}


def analyze_run(run, show=False):
    run, df, meta = _load(run)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    out = {"run": run.name, "kind": meta["kind"], "sim": meta.get("sim"), "land_reason": meta.get("land_reason")}
    sc = meta["scene"]
    P = np.array(sc["path_cm"])
    closed = sc["closed"]
    phases = df["phase"].astype(str)
    t = df["t"].to_numpy(float)
    ev = {e["event"]: e["t"] for e in meta.get("events", [])}
    t_take = next((e["t"] for e in meta["events"] if e["event"] == "phase -> TAKEOFF"), t[0])
    t_land = next((e["t"] for e in meta["events"] if e["event"] == "phase -> LAND"), t[-1])
    out["flight_time_s"] = float(t_land - t_take)
    trk = df[phases == "TRACK"]
    dep = depth_estimate(df, meta)
    if dep:
        out["depth"] = dep
    if len(trk) > 10:
        corners_s = corner_arclengths(P, sc["length_cm"]) if meta["kind"] == "square" else []
        tw = trk
        if corners_s:
            # step metrics only until 20 cm before the first corner after the start (corners are separate events)
            s0 = trk["s"].iloc[0]
            ahead = sorted(((c - s0) % sc["length_cm"]) for c in corners_s)
            first = next((a for a in ahead if a > 30), None)
            if first is not None:
                prog = (trk["s"] - s0) % sc["length_cm"]
                stop = np.argmax((prog > first - 20).to_numpy() & (prog < first + 50).to_numpy())
                if stop > 10:
                    tw = trk.iloc[:stop]
        m = step_metrics(tw["t"], tw["e_n"])
        steady = trk[trk["t"] >= trk["t"].iloc[0] + m.get("steady_from_s", 0)]["e_n"].to_numpy(float)
        steady = steady[np.isfinite(steady)]
        if len(steady) > 3:  # steady-state statistics over the whole rest of the lap
            m.update(ss_mean_cm=float(np.mean(steady)), ss_mean_abs_cm=float(np.mean(np.abs(steady))), ss_rms_cm=float(np.sqrt(np.mean(steady ** 2))),
                     ss_max_cm=float(np.max(np.abs(steady))), ss_p95_cm=float(np.percentile(np.abs(steady), 95)),
                     on_line_pct=float(100 * np.mean(np.abs(steady) <= 10.0)), on_white_pct=float(100 * np.mean(np.abs(steady) <= 2.0)))
        m["step_window_s"] = float(tw["t"].iloc[-1] - tw["t"].iloc[0])
        out["track"] = m
        tt = trk["t"].to_numpy(float)
        out["track"]["track_time_s"] = float(tt[-1] - tt[0])
        sv = trk["s"].to_numpy(float)
        out["track"]["mean_speed_cm_s"] = float(np.nansum(np.abs(np.diff(sv)) * (np.abs(np.diff(sv)) < 20)) / max(tt[-1] - tt[0], 1e-6))
        out["track"]["progress"] = float(np.nanmax(trk["progress"])) if "progress" in trk else np.nan
        if dep:
            k = dep["scale_to_drone_plane"]
            out["track"]["ss_rms_drone_plane_cm"] = out["track"].get("ss_rms_cm", np.nan) * k
        # corners of the square: worst error within +-25 cm of each corner
        if meta["kind"] == "square":
            out["corners"] = corner_metrics(trk, P, sc["length_cm"])
        fig_track(plt, run, trk, P, closed, m, meta)
    las = df[phases == "LASER"]
    if len(las) > 10:
        out["laser"] = laser_metrics(las)
        fig_laser(plt, run, df, P, meta)
    if (phases == "EXCITE").any():
        fig_sysid(plt, run, df)
    fig_overview(plt, run, df, meta)
    if meta["kind"] == "path":
        out["race_time_s"] = out["flight_time_s"]
    (run / "metrics.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (run / "metrics.md").write_text(metrics_md(out), encoding="utf-8")
    print(metrics_md(out))
    return out


def corner_arclengths(P, length):
    tan = np.diff(np.vstack([P, P[:1]]), axis=0)
    seglen = np.linalg.norm(tan, axis=1)
    s_v = np.concatenate([[0], np.cumsum(seglen)])[:-1]
    h = np.arctan2(tan[:, 1], tan[:, 0])
    dh = np.abs((h - np.roll(h, 1) + np.pi) % (2 * np.pi) - np.pi)
    return [float(c % length) for c in s_v[dh > np.radians(45)]]


def corner_metrics(trk, P, length):
    # corners = vertices with a heading change > 45 deg
    tan = np.diff(np.vstack([P, P[:1]]), axis=0)
    seglen = np.linalg.norm(tan, axis=1)
    s_v = np.concatenate([[0], np.cumsum(seglen)])[:-1]  # arclength of vertex i (start of segment i)
    h = np.arctan2(tan[:, 1], tan[:, 0])
    dh = np.abs((h - np.roll(h, 1) + np.pi) % (2 * np.pi) - np.pi)  # turn at vertex i (segment i-1 -> i)
    corners = [c % length for c in s_v[dh > np.radians(45)]]
    res = []
    s = trk["s"].to_numpy(float)
    e = trk["e_n"].to_numpy(float)
    for c in sorted(set(np.round(corners, 1))):
        d = (s - c + length / 2) % length - length / 2
        w = np.abs(d) < 25
        if w.any():
            res.append({"corner_s_cm": float(c), "max_abs_e_cm": float(np.nanmax(np.abs(e[w]))), "mean_e_cm": float(np.nanmean(e[w]))})
    return res


def laser_metrics(las):
    d = np.hypot(las["laserX"] - las["X"], las["laserY"] - las["Y"]).to_numpy(float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {}
    return {"laser_dist_mean_cm": float(np.mean(d)), "laser_dist_rms_cm": float(np.sqrt(np.mean(d ** 2))),
            "laser_dist_p95_cm": float(np.percentile(d, 95)), "laser_phase_s": float(las["t"].iloc[-1] - las["t"].iloc[0]),
            "laser_seen_pct": float(100 * np.isfinite(las["laserX"].to_numpy(float)).mean())}


def fig_track(plt, run, trk, P, closed, m, meta):
    nrm = np.zeros_like(P)
    tan = np.gradient(np.vstack([P[-2:], P, P[:2]]) if closed else P, axis=0)
    tan = tan[2:-2] if closed else tan
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    nrm[:, 0], nrm[:, 1] = -tan[:, 1], tan[:, 0]
    fig, ax = plt.subplots(figsize=(7, 7))
    PP = np.vstack([P, P[:1]]) if closed else P
    for off, st in ((10, "-"), (-10, "-")):
        Q = P + off * nrm
        Q = np.vstack([Q, Q[:1]]) if closed else Q
        ax.plot(Q[:, 0], Q[:, 1], st, color="#c8102e", lw=1, alpha=0.6)
    ax.plot(PP[:, 0], PP[:, 1], color="0.35", lw=2, label="ideal path (white centre line)")
    X, Y = trk["X"].to_numpy(float), trk["Y"].to_numpy(float)
    tt = trk["t"].to_numpy(float) - trk["t"].iloc[0]
    ts = m.get("steady_from_s", 0)
    tr = tt < ts
    ax.plot(X[tr], Y[tr], "--", color="tab:orange", lw=1.5, label="transient")
    ax.plot(X[~tr], Y[~tr], "-", color="tab:red", lw=1.5, label="steady state")
    ax.plot(X[0], Y[0], "o", color="k", label="start of tracking")
    ax.set_aspect("equal")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(f"{meta['kind']}: flight vs ideal path (red: line edges +-10 cm)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "fig_trajectory.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    e = trk["e_n"].to_numpy(float)
    axs[0].plot(tt, e, color="tab:blue", lw=1.2, label="e_n")
    axs[0].axhspan(-BAND, BAND, color="tab:green", alpha=0.12, label=f"+-{BAND:.0f} cm band")
    axs[0].axhline(0, color="k", lw=0.8)
    for k, c in (("settling_time_s", "tab:green"), ("peak_time_s", "tab:purple")):
        if np.isfinite(m.get(k, np.nan)):
            axs[0].axvline(m[k], color=c, ls="--", lw=1, label=f"{k} = {m[k]:.2f}")
    axs[0].set_ylabel("cross-track error (cm)")
    axs[0].legend(fontsize=8, loc="upper right")
    txt = ", ".join(f"{k}={m[k]:.2f}" for k in ("rise_time_s", "overshoot_pct", "ss_mean_cm", "ss_rms_cm") if k in m and np.isfinite(m[k]))
    axs[0].set_title(txt, fontsize=9)
    axs[1].plot(tt, trk["rc_a"], label="rc a (left/right)")
    axs[1].plot(tt, trk["rc_c"], label="rc c (up/down)")
    axs[1].set_ylabel("rc command")
    axs[1].set_xlabel("time since start of tracking (s)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "fig_error_time.png", dpi=150)
    plt.close(fig)


def fig_laser(plt, run, df, P, meta):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(P[:, 0], P[:, 1], color="0.6", lw=6, alpha=0.4, label="track")
    las = df[df["phase"] == "LASER"]
    ax.plot(las["laserX"], las["laserY"], ".", color="tab:green", ms=3, label="laser dot")
    ax.plot(df["X"], df["Y"], "-", color="tab:red", lw=1.2, label="drone")
    ob = meta["scene"].get("obstacle_cm")
    if ob:
        O = np.array(ob + ob[:1])
        ax.fill(O[:, 0], O[:, 1], color="k", alpha=0.7, label="obstacle")
    for b in meta["scene"].get("blue_cm", []):
        ax.add_patch(plt.Rectangle((b[0] - 10, b[1] - 10), 20, 20, color="tab:blue", alpha=0.6))
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("theme 2: path tracking + laser guidance")
    fig.tight_layout()
    fig.savefig(run / "fig_theme2.png", dpi=150)
    plt.close(fig)


def fig_sysid(plt, run, df):
    ex = df[df["phase"] == "EXCITE"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axs[0].plot(ex["t"], ex["rc_a"], label="rc a")
    axs[0].plot(ex["t"], ex["rc_c"], label="rc c")
    axs[0].legend()
    axs[0].set_ylabel("rc")
    axs[1].plot(ex["t"], ex["X"], label="X")
    axs[1].plot(ex["t"], ex["Y"], label="Y")
    axs[1].legend()
    axs[1].set_ylabel("cm")
    axs[1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(run / "fig_sysid_data.png", dpi=150)
    plt.close(fig)


def fig_overview(plt, run, df, meta):
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(df["t"], df["X"], label="X")
    axs[0].plot(df["t"], df["Y"], label="Y")
    axs[0].set_ylabel("position (cm)")
    axs[0].legend(fontsize=8)
    axs[1].plot(df["t"], df["rc_a"], label="a")
    axs[1].plot(df["t"], df["rc_c"], label="c")
    axs[1].set_ylabel("rc")
    axs[1].legend(fontsize=8)
    axs[2].plot(df["t"], df["vision_ms"], label="vision ms")
    axs[2].plot(df["t"], df["loop_ms"], label="loop ms")
    axs[2].set_ylabel("ms")
    axs[2].legend(fontsize=8)
    for ax in axs:
        for e in meta.get("events", []):
            if e["event"].startswith("phase ->"):
                ax.axvline(e["t"], color="0.6", lw=0.6, ls=":")
    for e in meta.get("events", []):
        if e["event"].startswith("phase ->"):
            axs[0].text(e["t"], axs[0].get_ylim()[1], e["event"][9:], fontsize=7, rotation=90, va="top")
    axs[2].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(run / "fig_overview.png", dpi=130)
    plt.close(fig)


def metrics_md(o):
    L = [f"# {o['run']}", "", f"- kind: {o['kind']}{' (SIM)' if o.get('sim') else ''}", f"- landed because: {o.get('land_reason')}",
         f"- flight time (take-off cmd -> land cmd): {o['flight_time_s']:.1f} s"]
    if "depth" in o:
        d = o["depth"]
        L.append(f"- drone apparent size {d['apparent_size_cm']:.1f} cm -> ~{d['drone_wall_dist_cm']:.0f} cm in front of the wall (wall->drone plane x{d['scale_to_drone_plane']:.3f})")
    if "track" in o:
        m = o["track"]
        L += ["", "## Path tracking (cross-track error e_n)", "", "| spec | value |", "|---|---|"]
        for k, u in (("e0_cm", "cm"), ("rise_time_s", "s"), ("overshoot_cm", "cm"), ("overshoot_pct", "%"), ("peak_time_s", "s"),
                     ("settling_time_s", "s (+-5 cm)"), ("settling_time_3cm_s", "s (+-3 cm)"), ("ss_mean_cm", "cm (steady-state error, signed)"),
                     ("ss_mean_abs_cm", "cm"), ("ss_rms_cm", "cm"), ("ss_max_cm", "cm"), ("ss_p95_cm", "cm"), ("on_line_pct", "% on red line"),
                     ("on_white_pct", "% within +-2 cm"), ("track_time_s", "s"), ("mean_speed_cm_s", "cm/s"), ("progress", "")):
            if k in m:
                v = m[k]
                L.append(f"| {k} | {v:.2f} {u} |" if isinstance(v, (int, float)) and np.isfinite(v) else f"| {k} | n/a |")
    if "corners" in o:
        L += ["", "corners: " + ", ".join(f"s={c['corner_s_cm']:.0f}: max {c['max_abs_e_cm']:.1f} cm" for c in o["corners"])]
    if "laser" in o:
        L += ["", "## Laser guidance", ""] + [f"- {k}: {v:.2f}" for k, v in o["laser"].items()]
    if "race_time_s" in o:
        L += ["", f"**race time: {o['race_time_s']:.1f} s**"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    for r in sys.argv[1:]:
        analyze_run(r)
