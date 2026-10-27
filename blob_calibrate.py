#!/usr/bin/env python
"""blob_calibrate.py -- pick blob_tracker.py thresholds from the video itself.

Why this exists
---------------
blob_tracker.py's gates were tuned on the brightly-lit sessions (C, F, H, K),
where the runnable surface is near-saturated and the animal is much darker than
it.  `surface_mask` keeps `bg >= --sat-thr` (default 245) and `candidates`
computes `(grey < --dark-thr) * mask`, so a detection can only ever be found
*inside* that mask.  On a dimmer video the surface never reaches 245, the mask
collapses to a few hundred saturated pixels of lamp or reflection, and the
tracker reports "0 candidates per frame" -- not weak detections, none at all.
No amount of --dark-thr tuning fixes that, because the blob has nowhere to live.

This script does not assume the surface is bright or the animal is dark.  It
finds the animal by *motion* -- pixels that depart from the background median --
and then reads the four gate values off the pixels the animal actually occupied:

    --sat-thr    low enough that the mask covers everywhere the animal was seen
    --dark-thr   above the animal's own grey values
    --min-depth  below its typical contrast against the local background
    --area       the 10th-90th percentile of its observed blob area

It then *tests* those values by calling the tracker's own surface_mask() and
candidates() over the sampled frames, so the recommendation comes with a
measured candidate rate rather than a promise.  Both polarities are tested: if
the animal is brighter than its surround (back-lighting, a white animal), the
dark-blob logic cannot work on the raw image and the script says so explicitly
instead of recommending thresholds that cannot help.

Outputs (prefix from --out-prefix, default alongside the video)
    <p>_calib.csv    the full parameter grid with measured candidate rates
    <p>_calib.png    background + suggested mask, motion occupancy, grey
                     histograms with the suggested cut lines, grid heatmap
    stdout           the ready-to-paste blob_tracker.py track arguments

Read the "RECOMMENDATION" block first.  If it reports no moving object, the
geometry rectangle or the clip is wrong and no threshold will help; if it
reports inverted polarity, the tracker needs an inversion option before these
videos can be tracked at all.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# same sibling-import reasoning as blob_diagnose.py: the whole point is to
# evaluate the gates the tracker itself applies, so these must be the tracker's
# own functions and not a reimplementation that could drift from them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blob_tracker as bt  # noqa: E402


# ------------------------------------------------------------------ helpers

def sample_indices(n_frames, n, window):
    if window:
        s, ln = window
        return np.linspace(s, min(n_frames - 1, s + ln - 1), n).astype(int)
    return np.linspace(0, n_frames - 1, n).astype(int)


def largest_blob(binary, elig, min_area=40):
    """Biggest connected component of `binary` inside `elig`, or None."""
    d = cv2.morphologyEx((binary * elig).astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    if nl < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < min_area:
        return None
    return lab == i


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


# -------------------------------------------------------------- calibration

def collect(cap, bg, elig, idx, noise_k):
    """Walk the sampled frames once, accumulating motion statistics."""
    # noise scale from the first few frames: the animal is a tiny fraction of
    # the eligible area, so the median |deviation| is dominated by sensor noise
    dev = []
    for f in idx[:8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dev.append(np.abs(g - bg)[elig > 0])
    sigma = 1.4826 * float(np.median(np.concatenate(dev))) if dev else 1.0
    thr = max(6.0, noise_k * sigma)

    acc = {"occ_dark": np.zeros(bg.shape, np.int32),
           "occ_bright": np.zeros(bg.shape, np.int32),
           "sigma": sigma, "thr": thr, "n_read": 0}
    for pol in ("dark", "bright"):
        acc[pol] = {"area": [], "depth": [], "grey": [], "bgval": [], "hits": 0}

    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if not ok:
            continue
        acc["n_read"] += 1
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = g - bg
        for pol, binary in (("dark", (diff < -thr)), ("bright", (diff > thr))):
            acc["occ_" + pol] += (binary.astype(np.int32) * elig)
            sel = largest_blob(binary.astype(np.uint8), elig)
            if sel is None:
                continue
            a = acc[pol]
            a["hits"] += 1
            a["area"].append(int(sel.sum()))
            a["depth"].append(float(np.abs(diff[sel]).mean()))
            # subsample pixel values so memory stays flat over long videos
            gv, bv = g[sel], bg[sel]
            step = max(1, len(gv) // 40)
            a["grey"].extend(gv[::step].tolist())
            a["bgval"].extend(bv[::step].tolist())
    return acc


def suggest(a, polarity):
    """Gate values read off the pixels the animal occupied.

    The tracker only knows how to find *dark* blobs on a *bright* surface, so a
    bright moving object is handled by inverting the image.  Every gate value is
    therefore derived on the scale the gates will actually be applied to --
    255 - grey when the object is bright -- not on the raw pixel values.
    """
    p = a[polarity]
    grey, bgv = np.array(p["grey"]), np.array(p["bgval"])
    if polarity == "bright":
        grey, bgv = 255.0 - grey, 255.0 - bgv
    # --area only has to exclude specks and the whole surface, so it is bracketed
    # generously around the observed sizes. Setting it at the observed percentiles
    # rejects the animal on the frames that defined them, and the motion mask and
    # candidates() apply morphology slightly differently, so a few px of slack is
    # needed even for a rigid object.
    s = {"n_blobs": p["hits"],
         "area_lo": int(max(30, np.floor(pct(p["area"], 5) * 0.6 / 10) * 10)),
         "area_hi": int(np.ceil(pct(p["area"], 95) * 1.8 / 10) * 10),
         "min_depth": float(max(5, np.floor(pct(p["depth"], 20)))),
         "sat_thr": float(np.floor(pct(bgv, 5))),
         "blob_grey_p90": pct(grey, 90),
         "blob_grey_med": pct(grey, 50),
         "surface_bg_med": pct(bgv, 50)}
    # the darkness cut must admit the animal's own greys with a little headroom,
    # and on the inverted scale the same logic applies to 255 - grey
    s["dark_thr"] = float(np.ceil(pct(grey, 90) + 0.25 * (pct(bgv, 50) - pct(grey, 90))))
    return s


def load_frames(cap, idx):
    """Decode the grid frames once -- a seek per grid cell is unaffordable here."""
    out = []
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, im = cap.read()
        if ok:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return out


def evaluate(frames, bg, geom, sat_thr, dark_thr, min_depth, area, invert):
    """Candidate rate the tracker's own gates would give on the sampled frames."""
    b = (255.0 - bg) if invert else bg
    m = bt.surface_mask(b, geom, sat_thr, 0)
    if m.sum() < 50:
        return {"mask_px": int(m.sum()), "cand_rate": 0.0, "area_med": float("nan"),
                "depth_med": float("nan"), "n_cand_med": 0.0}
    nc, areas, depths = [], [], []
    for fr in frames:
        g = fr.astype(np.float32)
        if invert:
            g = 255.0 - g
        cands, _, _ = bt.candidates(g, b, m, dark_thr, area[0], area[1], min_depth)
        nc.append(len(cands))
        if cands:
            best = max(cands, key=lambda c: c["score"])
            areas.append(best["area"]); depths.append(best["depth"])
    return {"mask_px": int(m.sum()),
            "cand_rate": float(np.mean([n > 0 for n in nc])) if nc else 0.0,
            "n_cand_med": float(np.median(nc)) if nc else 0.0,
            "area_med": pct(areas, 50), "depth_med": pct(depths, 50)}


# ------------------------------------------------------------------ figure

def figure(path, bg, geom, acc, elig, sug, pol, grid, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    m = bt.surface_mask((255.0 - bg) if pol == "bright" else bg, geom, best["sat_thr"], 0)
    vis = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    vis[m > 0] = (0.55 * vis[m > 0] + np.array([0, 90, 0])).astype(np.uint8)
    ax[0, 0].imshow(vis)
    ax[0, 0].set_title(f"recommended mask at --sat-thr {best['sat_thr']:.0f}\n"
                       f"{int(m.sum())} px, {100*m.sum()/max(1,elig.sum()):.0f}% of eligible region")

    occ = acc["occ_dark"] if pol == "dark" else acc["occ_bright"]
    im = ax[0, 1].imshow(np.ma.masked_equal(occ, 0), cmap="magma",
                         norm=LogNorm(vmin=1, vmax=max(2, occ.max())))
    ax[0, 1].set_title(f"where the {pol} moving object was seen\n"
                       f"({acc[pol]['hits']}/{acc['n_read']} sampled frames)")
    fig.colorbar(im, ax=ax[0, 1], fraction=.046, label="sampled frames")

    gv, bv = np.array(acc[pol]["grey"]), np.array(acc[pol]["bgval"])
    if pol == "bright":
        gv, bv = 255.0 - gv, 255.0 - bv
    # shared fixed bins: with per-call auto-binning a narrow distribution is
    # spread over 60 sub-pixel-wide bars and renders as nothing at all
    bins = np.arange(0, 258, 4)
    ax[1, 0].hist(bv, bins=bins, color="0.6", label="background under the animal")
    ax[1, 0].hist(gv, bins=bins, color="#1f77b4", alpha=.8, label="animal pixels")
    ax[1, 0].axvline(best["dark_thr"], color="#d62728", lw=1.4, label="--dark-thr")
    ax[1, 0].axvline(best["sat_thr"], color="#2ca02c", lw=1.4, ls="--", label="--sat-thr")
    ax[1, 0].set(xlabel="grey level" + (" (inverted: 255 - grey)" if pol == "bright" else ""),
                 ylabel="pixels", title="Animal vs its local background", xlim=(0, 255))
    ax[1, 0].legend(frameon=False, fontsize=7)

    sats = sorted({r["sat_thr"] for r in grid})
    darks = sorted({r["dark_thr"] for r in grid})
    Z = np.full((len(darks), len(sats)), np.nan)
    for r in grid:
        if r["min_depth"] == best["min_depth"] and r["invert"] == best["invert"]:
            Z[darks.index(r["dark_thr"]), sats.index(r["sat_thr"])] = r["cand_rate"]
    im = ax[1, 1].imshow(Z, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax[1, 1].set(xticks=range(len(sats)), xticklabels=[f"{s:.0f}" for s in sats],
                 yticks=range(len(darks)), yticklabels=[f"{d:.0f}" for d in darks],
                 xlabel="--sat-thr", ylabel="--dark-thr",
                 title=f"frames with >=1 candidate\n(--min-depth {best['min_depth']:.0f})")
    for i in range(len(darks)):
        for j in range(len(sats)):
            if np.isfinite(Z[i, j]):
                ax[1, 1].text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                              fontsize=7, color="w" if Z[i, j] < .6 else "k")
    fig.colorbar(im, ax=ax[1, 1], fraction=.046)
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# -------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--geom", required=True)
    p.add_argument("--bg", default=None, help="cached *_mask_bg.npy from the mask step")
    p.add_argument("--bg-stride", type=int, default=900)
    p.add_argument("--n", type=int, default=240, help="frames sampled for calibration")
    p.add_argument("--grid-n", type=int, default=60, help="frames used per grid cell")
    p.add_argument("--window", type=int, nargs=2, default=None, metavar=("START", "LEN"))
    p.add_argument("--noise-k", type=float, default=4.0,
                   help="motion threshold in robust sigmas of the frame-to-background deviation")
    p.add_argument("--out-prefix", default=None)
    a = p.parse_args()

    geom = json.load(open(a.geom))
    cap, N = bt.open_video(a.video)
    bg = (np.load(a.bg).astype(np.float32) if a.bg and os.path.exists(a.bg)
          else bt.background(cap, N, a.bg_stride))
    # the tracker's own definition of the eligible region, so the percentages
    # here and the ones cmd_mask prints are the same number
    elig = bt.search_region(bg.shape, geom)
    pre = a.out_prefix or os.path.splitext(a.video)[0]
    print(f"frame {bg.shape[1]}x{bg.shape[0]}, {N} frames; "
          f"eligible region {int(elig.sum())} px")
    print(f"eligible-region background percentiles (p5,p25,p50,p75,p95): "
          + ", ".join(f"{np.percentile(bg[elig > 0], q):.0f}" for q in (5, 25, 50, 75, 95)))

    idx = sample_indices(N, a.n, a.window)
    acc = collect(cap, bg, elig, idx, a.noise_k)
    print(f"motion threshold {acc['thr']:.1f} grey levels "
          f"(robust sigma {acc['sigma']:.2f}); read {acc['n_read']} frames")
    for pol in ("dark", "bright"):
        print(f"  {pol:>6} blobs found in {acc[pol]['hits']}/{acc['n_read']} frames"
              + (f", area median {pct(acc[pol]['area'], 50):.0f} px, "
                 f"contrast median {pct(acc[pol]['depth'], 50):.0f} grey levels"
                 if acc[pol]["hits"] else ""))

    if max(acc["dark"]["hits"], acc["bright"]["hits"]) < 0.15 * max(1, acc["n_read"]):
        print("\nRECOMMENDATION: no moving object found in the eligible region.\n"
              "  Nothing moved against the background in these frames, so no threshold\n"
              "  can help. Either the maze_outer rectangle in the geometry file does not\n"
              "  cover the surface, or the animal is not on the maze in this clip.\n"
              "  Check the mask PNG against the frame before tuning anything.")
        return

    pol = "dark" if acc["dark"]["hits"] >= acc["bright"]["hits"] else "bright"
    sug = suggest(acc, pol)

    gidx = idx[:: max(1, len(idx) // a.grid_n)]
    frames = load_frames(cap, gidx)
    sats = sorted({245.0, sug["sat_thr"], max(1.0, sug["sat_thr"] - 15), sug["sat_thr"] + 15})
    darks = sorted({200.0, sug["dark_thr"], max(1.0, sug["dark_thr"] - 20), sug["dark_thr"] + 20})
    area = [sug["area_lo"], sug["area_hi"]]
    grid = []
    for inv in (False, True):
        for md in sorted({sug["min_depth"], 40.0}):
            for st in sats:
                for dt in darks:
                    r = evaluate(frames, bg, geom, st, dt, md, area, inv)
                    r.update(sat_thr=st, dark_thr=dt, min_depth=md, invert=int(inv),
                             area_lo=area[0], area_hi=area[1])
                    grid.append(r)
    with open(pre + "_calib.csv", "w") as fh:
        keys = ["invert", "sat_thr", "dark_thr", "min_depth", "area_lo", "area_hi",
                "mask_px", "cand_rate", "n_cand_med", "area_med", "depth_med"]
        fh.write(",".join(keys) + "\n")
        for r in grid:
            fh.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"wrote {pre}_calib.csv")

    best = max(grid, key=lambda r: (r["cand_rate"], -r["n_cand_med"]))
    dflt = next(r for r in grid if r["sat_thr"] == 245.0 and r["dark_thr"] == 200.0
                and r["min_depth"] == 40.0 and r["invert"] == 0)
    figure(pre + "_calib.png", bg, geom, acc, elig, sug, pol, grid, best)

    print(f"\nRECOMMENDATION ({pol} moving object)")
    print(f"  current defaults  : candidate rate {dflt['cand_rate']:.2f} "
          f"(mask {dflt['mask_px']} px)")
    print(f"  best on this grid : candidate rate {best['cand_rate']:.2f} "
          f"(mask {best['mask_px']} px, median area {best['area_med']:.0f} px, "
          f"median depth {best['depth_med']:.0f})")
    if best["invert"]:
        print("\n  The animal is BRIGHTER than its surround here, so the dark-blob gates\n"
              "  cannot work on the raw image -- the winning grid cell needed the image\n"
              "  inverted. blob_tracker.py has no --invert option; it needs one before\n"
              "  these videos can be tracked. Do not use the numbers below as they stand.")
    elif best["cand_rate"] < 0.5:
        print("\n  Under half the sampled frames yield a candidate even at the best grid\n"
              "  cell. The gates are not the limiting factor -- inspect the occupancy\n"
              "  panel and the mask PNG before tracking.")
    print("\n  blob_tracker.py track \\\n"
          f"    --sat-thr {best['sat_thr']:.0f} --dark-thr {best['dark_thr']:.0f} \\\n"
          f"    --min-depth {best['min_depth']:.0f} "
          f"--area {area[0]} {area[1]}")
    print("\n  Verify on a window before the full run; re-run the mask step with the new\n"
          "  --sat-thr first, since the cached *_mask_bg.npy is reused but the mask is not.")


if __name__ == "__main__":
    main()
