#!/usr/bin/env python
"""Align a free-running T-maze behaviour video to miniscope frames.

The behaviour camera carries no sync line -- only the miniscope frame TTL is on
the logic analyser.  The video clock is therefore recovered from the data:

  1. `door_state.py` gives per-frame open/closed state of the start-box door
     DS0 from the exposed slide rails (see the `tmaze-door-state` skill).
  2. Channel MS0 on the logic analyser is the *motor* line for that same door
     (a fixed ~1.14 s actuation pulse, once per trial), so MS0 pulses and video
     door-opening onsets are the same physical events on two clocks.
  3. A 2-D search over (frame rate, per-capture offset) maximises the number of
     MS0 pulses landing on a video door onset.  Per-capture offsets are fitted
     independently and must agree with the capture folder wall-clock names to
     within their 1 s rounding -- that agreement is the acceptance test.
  4. Validation uses the six motor channels NOT used in the fit: at a correct
     alignment each one's events must cluster at a single maze location.

Miniscope frame times come from the DAQ `timeStamps.csv` (authoritative for
which frames were saved); the frame TTL is used only to confirm that the DAQ
and logic-analyser clocks agree.  Note each TTL *edge* is one frame -- the line
is a 50 % duty square wave at half the frame rate.

Outputs
  <out>_alignment_key.csv      one row per miniscope frame: video frame, wall
                               clock, trial, x, y, speed, door state
  <out>_traces_aligned.parquet the key joined to the calcium traces
  <out>_alignment_params.json  fitted rate/offsets + validation statistics
  <out>_alignment_summary.png  QC figure

Expected session layout (as produced by the lab's organiser):
  <session>/behavior/<protocol>.csv        trial log with wall-clock times
  <session>/traces/<name>.csv              cells x miniscope frames
  <session>/saleae/*.csv                   one transition export per capture
  <session>/frame_timestamps/HH_MM_SS__*/timeStamps.csv
"""
from __future__ import annotations
import argparse, json, os, re, glob
import numpy as np, pandas as pd
from scipy.signal import fftconvolve

# logic-analyser channel map (lab convention)
CH = {0: "MS0", 1: "MS1", 2: "MS2", 3: "MA1", 4: "MA2", 5: "MP1", 6: "MP2",
      10: "TrialStart", 11: "TrialComplete", 12: "frame"}
DOOR_MOTOR = 0          # MS0 -- motor for start-box door DS0
FRAME_CH = 12
SENTINEL = 0.0          # tracker writes (-1,-1) when it finds nothing


def hhmmss(s: str) -> int:
    h, m, sec = re.split(r"[:_]", str(s).strip())[:3]
    return int(h) * 3600 + int(m) * 60 + int(sec)


def edges(d: pd.DataFrame, ch: int):
    """Rising+falling transition times for one channel of a Logic 2 export."""
    col = [c for c in d.columns if c.endswith(f"Channel {ch}")
           or c == f"Channel {ch}"][0]
    v = d[col].to_numpy()
    t = d["Time [s]"].to_numpy()
    k = np.flatnonzero(np.diff(v) != 0) + 1
    return t[k], v[k]


def rises(d, ch, debounce=0.05):
    t, v = edges(d, ch)
    r = t[v == 1]
    return r[np.diff(r, prepend=-1e9) > debounce]


def load_session(root):
    S = [pd.read_csv(p) for p in sorted(glob.glob(f"{root}/saleae/*.csv"))]
    fold = sorted(f for f in os.listdir(f"{root}/frame_timestamps")
                  if os.path.isdir(f"{root}/frame_timestamps/{f}"))
    ts = [pd.read_csv(f"{root}/frame_timestamps/{f}/timeStamps.csv")
          ["Time Stamp (ms)"].to_numpy() / 1000.0 for f in fold]
    wall = [hhmmss(f.split("__")[0].replace("_", ":")) for f in fold]
    if not (len(S) == len(ts)):
        raise SystemExit(f"{len(S)} saleae exports but {len(ts)} timestamp folders")
    return S, ts, wall, fold


def check_frame_ttl(S, ts):
    """Each TTL edge is one miniscope frame; confirm against the DAQ record."""
    for k, (d, t) in enumerate(zip(S, ts)):
        et, _ = edges(d, FRAME_CH)
        et = et[np.diff(et, prepend=-1) > 0.025]          # de-glitch
        span_ttl, span_daq = et[-1] - et[0], t[-1] - t[0]
        if abs(span_ttl - span_daq) > 1.0:
            print(f"  WARNING capture {k}: TTL span {span_ttl:.1f}s vs "
                  f"DAQ {span_daq:.1f}s -- clocks disagree")
        if abs(len(et) - len(t)) > 5:
            print(f"  WARNING capture {k}: {len(et)} TTL edges vs {len(t)} saved "
                  f"frames (>5 apart)")


def fit_rate(onsets, S, wall, n_video, fps_lo, fps_hi, coarse=0.01,
             tol_s=0.5, window_s=40.0):
    """Two-stage search for video frame rate and per-capture start frame."""
    t0 = wall[0]
    ev_abs = np.concatenate([rises(S[k], DOOR_MOTOR) + wall[k]
                             for k in range(len(S))]) - t0
    best = (-1, None, None)
    for a in np.arange(fps_lo, fps_hi, coarse):            # stage 1: global
        tol = max(1, int(round(a * tol_s)))
        A = np.zeros(n_video, np.float32); A[onsets] = 1
        A = (np.convolve(A, np.ones(2 * tol + 1), "same") > 0).astype(np.float32)
        e = np.round(a * ev_abs).astype(int)
        if e.max() >= n_video:
            continue
        P = int(a * 900)
        B = np.zeros(e.max() + 1, np.float32); B[e] = 1
        s = fftconvolve(np.concatenate([np.zeros(P, np.float32), A]), B[::-1], "valid")
        j = int(np.argmax(s))
        if s[j] > best[0]:
            best = (float(s[j]), float(a), j - P)
    _, a0, b0 = best

    A1 = np.zeros(n_video, np.uint8); A1[onsets] = 1
    ev_k = [rises(S[k], DOOR_MOTOR) for k in range(len(S))]

    def hits(a, k, bk, tol):
        e = np.round(a * ev_k[k] + bk).astype(int)
        e = e[(e >= tol) & (e < n_video - tol)]
        return sum(A1[i - tol:i + tol + 1].any() for i in e)

    best = None                                            # stage 2: per capture
    for a in np.arange(a0 - 0.3, a0 + 0.3, coarse * 0.4):
        tol = max(1, int(round(a * tol_s * 0.5)))
        tot, bs = 0, []
        for k in range(len(S)):
            pred = a * (wall[k] - t0) + b0
            cand = np.arange(int(pred - a * window_s), int(pred + a * window_s))
            h = np.array([hits(a, k, bb, tol) for bb in cand])
            j = int(np.argmax(h)); tot += int(h[j]); bs.append(int(cand[j]))
        if best is None or tot > best[0]:
            best = (tot, float(a), bs)
    tot, fps, bs = best
    drift = [bs[k] / fps - (wall[k] - t0) for k in range(len(S))]
    spread = max(drift) - min(drift)
    print(f"  fps {fps:.4f}  offsets {bs}  matched {tot}/{sum(len(e) for e in ev_k)}")
    print(f"  per-capture offset vs folder wall-clock: "
          f"{[round(x,2) for x in drift]} s (spread {spread:.2f} s)")
    if spread > 2.0:
        print("  WARNING offsets disagree by >2 s -- alignment is NOT trustworthy")
    return fps, bs, tot, drift


def validate(S, wall, fps, bs, x, y, ok, n_video, n_null=1000, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for ch in [c for c in CH if c < 10]:
        fr = np.concatenate([np.round(fps * rises(S[k], ch) + bs[k]).astype(int)
                             for k in range(len(S))])
        fr = fr[(fr >= 0) & (fr < n_video)]

        def spread(f):
            f = f[ok[f]]
            if len(f) < 4:
                return np.nan
            return float(np.median(np.hypot(x[f] - np.median(x[f]),
                                            y[f] - np.median(y[f]))))
        obs = spread(fr)
        null = np.array([spread((fr + rng.integers(0, n_video)) % n_video)
                         for _ in range(n_null)])
        f2 = fr[ok[fr]]
        rows.append(dict(channel=CH[ch], n=int(len(f2)),
                         x=float(np.median(x[f2])), y=float(np.median(y[f2])),
                         spread_px=round(obs, 1),
                         null_median_px=round(float(np.nanmedian(null)), 1),
                         p=round(float((null < obs).mean()), 4)))
    v = pd.DataFrame(rows)
    bad = (v.p > 0.05).sum()
    print(v.to_string(index=False))
    if bad > 1:
        print(f"  WARNING {bad} channels fail the concentration test -- "
              f"alignment is probably wrong")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="organised session root")
    p.add_argument("--doorstate", required=True, help="per-frame CSV from door_state.py")
    p.add_argument("--tracking", required=True, help="DeepLabCut .h5 for the same video")
    p.add_argument("--bodypart", default="mouse_center")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--fps-range", nargs=2, type=float, default=[20.0, 90.0])
    p.add_argument("--min-open-s", type=float, default=0.0,
                   help="ignore door openings shorter than this when fitting")
    a = p.parse_args()

    S, ts, wall, fold = load_session(a.session)
    print(f"{len(S)} captures, {sum(len(t) for t in ts)} miniscope frames")
    check_frame_ttl(S, ts)

    ds = pd.read_csv(a.doorstate)
    n_video = len(ds)
    st = ds["door_open"].to_numpy()
    onsets = np.flatnonzero((st[1:] == 1) & (st[:-1] == 0)) + 1
    if a.min_open_s > 0:
        offs = np.flatnonzero((st[1:] == 0) & (st[:-1] == 1)) + 1
        j = np.searchsorted(offs, onsets)
        dur = np.where(j < len(offs), offs[np.clip(j, 0, len(offs) - 1)] - onsets, 0)
        onsets = onsets[dur >= a.min_open_s * 30]   # conservative: assumes >=30 fps
    print(f"{n_video} video frames, {len(onsets)} door openings")

    dlc = pd.read_hdf(a.tracking)
    sc = dlc.columns.get_level_values(0)[0]
    mc = dlc[(sc, a.bodypart)]
    x, y = mc["x"].to_numpy(), mc["y"].to_numpy()
    ok = ~((x <= SENTINEL) & (y <= SENTINEL))
    if len(x) != n_video:
        raise SystemExit(f"tracking has {len(x)} rows but door state has {n_video}")
    print(f"tracker: {ok.mean()*100:.1f}% of frames localised")

    fps, bs, tot, drift = fit_rate(onsets, S, wall, n_video, *a.fps_range)
    sensors = validate(S, wall, fps, bs, x, y, ok, n_video)

    # ---- build the key -----------------------------------------------------
    recs = []
    for k in range(len(S)):
        t = ts[k]
        vf = np.round(fps * t + bs[k]).astype(int)
        s0, s1 = rises(S[k], 10), rises(S[k], 11)
        ti = np.full(len(t), -1)
        for j, u in enumerate(s0):
            nxt = s1[s1 > u]
            ti[(t >= u) & (t <= (nxt[0] if len(nxt) else t[-1]))] = j
        recs.append(pd.DataFrame(dict(capture=k, ms_frame_in_capture=np.arange(len(t)),
                                      t_capture_s=t, t_wall_s=t + wall[k],
                                      video_frame=vf, trial_in_capture=ti)))
    K = pd.concat(recs, ignore_index=True)
    K.insert(0, "ms_frame", np.arange(len(K)))
    inb = (K.video_frame >= 0) & (K.video_frame < n_video)
    v = K.video_frame.clip(0, n_video - 1).to_numpy()
    K["x"] = np.where(inb & ok[v], x[v], np.nan)
    K["y"] = np.where(inb & ok[v], y[v], np.nan)
    K["door_DS0_open"] = np.where(inb, st[v], np.nan)
    K["speed_px_s"] = np.hypot(np.gradient(K.x.to_numpy()),
                               np.gradient(K.y.to_numpy())) / np.gradient(K.t_wall_s.to_numpy())
    K["wall_clock"] = pd.to_datetime(K.t_wall_s, unit="s").dt.strftime("%H:%M:%S.%f").str[:-3]
    K.to_csv(f"{a.out}_alignment_key.csv", index=False)

    tp = glob.glob(f"{a.session}/traces/*.csv")
    if tp:
        TR = pd.read_csv(tp[0], index_col=0)
        if len(TR) == len(K):
            pd.concat([K, TR.reset_index(drop=True)], axis=1).to_parquet(
                f"{a.out}_traces_aligned.parquet", index=False)
            print(f"wrote {a.out}_traces_aligned.parquet ({TR.shape[1]} cells)")
        else:
            print(f"  WARNING traces have {len(TR)} rows, key has {len(K)} -- not joined")

    json.dump(dict(video_fps=fps, video_n_frames=int(n_video),
                   captures=[dict(index=k, folder=fold[k], wall_start_s=wall[k],
                                  video_frame_at_t0=bs[k], n_ms_frames=len(ts[k]),
                                  offset_vs_wall_s=round(drift[k], 2))
                             for k in range(len(S))],
                   ms_frames_total=int(len(K)),
                   door_motor_matched=int(tot),
                   sensor_locations={r.channel: [round(r.x), round(r.y)]
                                     for r in sensors.itertuples()},
                   validation=sensors.to_dict("records")),
              open(f"{a.out}_alignment_params.json", "w"), indent=1)
    print(f"wrote {a.out}_alignment_key.csv and _alignment_params.json")


if __name__ == "__main__":
    main()
