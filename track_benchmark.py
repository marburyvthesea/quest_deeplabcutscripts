#!/usr/bin/env python
"""Score DeepLabCut and the blob tracker against the same human labels.

Both estimate one number per frame -- where the animal is -- so they can be
compared directly, but only if three things are handled honestly:

  1. SPLIT.  The network was trained on most of the labelled frames; the blob
     tracker has never seen a label.  Error pooled over all labelled frames
     therefore flatters DLC.  Pass --split-doc (the project's
     Documentation_data-*.pickle) and train/test frames are reported apart.
     Without it the script still runs and says loudly that DLC is scored on
     frames it memorised.

  2. BIAS.  A human clicks an anatomical point (body_center); the blob tracker
     returns the centroid of the silhouette, which sits toward the body mass
     and shifts with tail and posture.  A constant offset between the two is
     a calibration constant, not an error -- align_tmaze.py cares about where
     the animal is frame to frame, and a fixed shift is removed by the maze
     geometry.  Both raw and bias-corrected (median offset subtracted) error
     are reported; the bias-corrected scatter is the number that matters.

  3. COVERAGE.  Accuracy on frames where a method reports a position says
     nothing about frames where it reports none.  A tracker that is accurate
     on 40% of frames is worse than one slightly less accurate on 99%.
     Coverage over the whole session is reported next to accuracy.

Outputs
  <out>_errors.csv     one row per labelled frame, both methods, split tagged
  <out>_summary.json   the numbers quoted below, machine-readable
  <out>_benchmark.png  error CDF, offset scatter, coverage, method agreement
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OCC_BIN = 8   # px; occupancy bin, ~1/3 of the animal's width in these frames


def frame_of(name) -> int | None:
    """Frame number from a labelled-image path (img0001937.png -> 1937)."""
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else None


def load_labels(path, animal, bodypart):
    """Human labels -> DataFrame[frame, xh, yh], restricted to one animal."""
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "CollectedData*.h5"),
                                recursive=True))
        if not hits:
            raise SystemExit(f"no CollectedData*.h5 under {path}")
        if animal:
            keep = [h for h in hits if f"{os.sep}{animal}{os.sep}" in h]
            if not keep:
                # Falling back to every animal's labels here silently pools
                # unrelated recordings and compares them to one video's
                # predictions -- random pairing that looks like a real but
                # enormous error (it put H at ~280 px from every part, with
                # the spread of a uniform draw over the arena). An animal
                # with no labels has to be an error, not a guess.
                have = sorted({os.path.basename(os.path.dirname(h))
                               for h in hits})
                raise SystemExit(
                    f"no labeled-data/{animal}/ under {path}; "
                    f"labelled animals are {have}")
            hits = keep
        df = pd.concat([pd.read_hdf(h) for h in hits])
    else:
        df = pd.read_hdf(path)

    # columns are (scorer, bodypart, coords); one scorer per project
    scorer = df.columns.get_level_values(0)[0]
    parts = set(df.columns.get_level_values(1))
    if bodypart not in parts:
        raise SystemExit(f"bodypart {bodypart!r} not in labels; have {sorted(parts)}")

    # the index is the image path: ('labeled-data', <animal>, 'imgNNNN.png')
    idx = df.index
    paths = ([os.path.join(*t) for t in idx] if isinstance(idx, pd.MultiIndex)
             else [str(i) for i in idx])
    out = pd.DataFrame({
        "frame": [frame_of(p) for p in paths],
        "src": paths,
        "xh": df[(scorer, bodypart, "x")].to_numpy(float),
        "yh": df[(scorer, bodypart, "y")].to_numpy(float)})
    if animal:
        sel = out.src.str.contains(rf"(?:^|{re.escape(os.sep)}){re.escape(animal)}"
                                   rf"(?:{re.escape(os.sep)}|$)", regex=True)
        if sel.any():
            out = out[sel]
    out = out.dropna(subset=["frame", "xh", "yh"])
    out["frame"] = out.frame.astype(int)
    return out.drop_duplicates("frame").set_index("frame").sort_index()


def load_dlc(path, bodypart, pcutoff):
    """Video predictions -> DataFrame[frame, xd, yd, p], NaN below pcutoff."""
    df = pd.read_hdf(path)
    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        parts = set(df.columns.get_level_values(1))
        if bodypart not in parts:
            raise SystemExit(f"bodypart {bodypart!r} not in {path}; have {sorted(parts)}")
        sub = df[scorer][bodypart]
    else:
        sub = df
    out = pd.DataFrame({"xd": sub["x"].to_numpy(float),
                        "yd": sub["y"].to_numpy(float),
                        "p": sub["likelihood"].to_numpy(float)},
                       index=np.arange(len(sub)))
    out.index.name = "frame"
    low = out.p < pcutoff
    out.loc[low, ["xd", "yd"]] = np.nan
    return out


def load_blob(path, drop_interp):
    """blob_tracker cmd_track CSV -> DataFrame[frame, xb, yb]."""
    T = pd.read_csv(path)
    out = pd.DataFrame({"xb": T.x.to_numpy(float), "yb": T.y.to_numpy(float)},
                       index=T.frame.astype(int))
    out.index.name = "frame"
    if drop_interp and "interp" in T:
        out.loc[T.interp.to_numpy(bool), ["xb", "yb"]] = np.nan
    return out


def load_split(doc, labels):
    """Map labelled frames to 'train'/'test' using DLC's own split pickle."""
    with open(doc, "rb") as fh:
        obj = pickle.load(fh)
    data, train_idx, test_idx = obj[0], np.asarray(obj[1]), np.asarray(obj[2])
    # `data` is the shuffle's image list, in the order the indices refer to
    names = []
    for rec in data:
        if isinstance(rec, dict):
            names.append(rec.get("image") or rec.get("image_path") or "")
        elif isinstance(rec, (list, tuple)) and rec:
            names.append(rec[0])
        else:
            names.append(str(rec))
    frames = np.array([frame_of(n) if frame_of(n) is not None else -1 for n in names])
    tag = pd.Series("unused", index=labels.index, dtype=object)
    for which, ids in (("train", train_idx), ("test", test_idx)):
        f = frames[ids[(ids >= 0) & (ids < len(frames))]]
        tag[tag.index.isin(f)] = which
    return tag


def stats(dx, dy, label):
    """Raw and bias-corrected error summary for one method."""
    ok = np.isfinite(dx) & np.isfinite(dy)
    n = int(ok.sum())
    if n == 0:
        return {"method": label, "n": 0}
    dx, dy = dx[ok], dy[ok]
    r = np.hypot(dx, dy)
    bx, by = np.median(dx), np.median(dy)
    rc = np.hypot(dx - bx, dy - by)
    return {"method": label, "n": n,
            "median_px": float(np.median(r)),
            "rmse_px": float(np.sqrt(np.mean(r ** 2))),
            "p90_px": float(np.percentile(r, 90)),
            "bias_x_px": float(bx), "bias_y_px": float(by),
            "median_debiased_px": float(np.median(rc)),
            "rmse_debiased_px": float(np.sqrt(np.mean(rc ** 2))),
            "p90_debiased_px": float(np.percentile(rc, 90))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=None,
                   help="CollectedData_*.h5, or a project/labeled-data dir to "
                        "search. Omit for a video with no labels: you then get "
                        "coverage and occupancy only, no accuracy.")
    p.add_argument("--dlc", required=True, help="DLC .h5 for the whole video")
    p.add_argument("--blob", default=None,
                   help="blob_tracker cmd_track CSV. Omit to assess DLC alone.")
    p.add_argument("--animal", default=None, help="restrict labels to this folder")
    p.add_argument("--bodypart", default="body_center")
    p.add_argument("--pcutoff", type=float, default=0.6)
    p.add_argument("--split-doc", default=None,
                   help="training-datasets/.../Documentation_data-*.pickle")
    p.add_argument("--count-interpolated", action="store_true",
                   help="treat gap-filled blob frames as real detections")
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args()

    dlc = load_dlc(a.dlc, a.bodypart, a.pcutoff)
    # an absent input becomes an all-NaN column rather than a branch: every
    # statistic below already skips NaN, so the DLC-only case needs no
    # separate code path and cannot drift from the three-way one
    lab = (load_labels(a.labels, a.animal, a.bodypart) if a.labels else
           pd.DataFrame({"src": [], "xh": [], "yh": []},
                        index=pd.Index([], name="frame", dtype=int)))
    blob = (load_blob(a.blob, not a.count_interpolated) if a.blob else
            pd.DataFrame({"xb": np.nan, "yb": np.nan}, index=dlc.index))
    print(f"{len(lab)} labelled frames, {len(dlc)} DLC frames, "
          f"{int(np.isfinite(blob.xb).sum())} blob detections")
    if not a.labels:
        print("no --labels: reporting coverage and occupancy only. A plausible\n"
              "  occupancy map is necessary but not sufficient -- it rules out a\n"
              "  stuck tracker, not a systematically displaced one.")

    J = lab.join(dlc, how="left").join(blob, how="left")
    if a.split_doc:
        J["split"] = load_split(a.split_doc, lab)
    else:
        J["split"] = "unknown"
    if a.labels and not a.split_doc:
        print("WARNING: no --split-doc, so most of these labelled frames are ones\n"
              "         the network trained on. Its error here is optimistic and\n"
              "         not comparable to the blob tracker, which saw no labels.")
    J["dlc_err"] = np.hypot(J.xd - J.xh, J.yd - J.yh)
    J["blob_err"] = np.hypot(J.xb - J.xh, J.yb - J.yh)

    rows = []
    for split in ("test", "train", "unused", "unknown"):
        m = J.split == split
        if not m.any():
            continue
        for meth, (dx, dy) in (("dlc", (J.xd[m] - J.xh[m], J.yd[m] - J.yh[m])),
                               ("blob", (J.xb[m] - J.xh[m], J.yb[m] - J.yh[m]))):
            s = stats(dx.to_numpy(), dy.to_numpy(), meth)
            s["split"] = split
            s["n_labelled"] = int(m.sum())
            rows.append(s)

    # coverage over the whole session, not just labelled frames
    both = dlc.join(blob, how="outer")
    cov = {"n_frames": int(len(both)),
           "dlc_frac": float(np.isfinite(both.xd).mean()),
           "blob_frac": float(np.isfinite(both.xb).mean()),
           "either_frac": float((np.isfinite(both.xd) | np.isfinite(both.xb)).mean()),
           "both_frac": float((np.isfinite(both.xd) & np.isfinite(both.xb)).mean())}
    agree = np.hypot(both.xd - both.xb, both.yd - both.yb)
    ok = np.isfinite(agree)
    if ok.any():
        cov["agreement_median_px"] = float(np.median(agree[ok]))
        cov["agreement_p90_px"] = float(np.percentile(agree[ok], 90))
        ddx, ddy = (both.xd - both.xb)[ok], (both.yd - both.yb)[ok]
        resid = np.hypot(ddx - np.median(ddx), ddy - np.median(ddy))
        cov["agreement_offset_x_px"] = float(np.median(ddx))
        cov["agreement_offset_y_px"] = float(np.median(ddy))
        cov["agreement_median_debiased_px"] = float(np.median(resid))
        cov["agreement_p90_debiased_px"] = float(np.percentile(resid, 90))

    # A tracker locked onto fixed rig hardware looks confident and never moves.
    # These two numbers say so with no ground truth to compare against, which
    # is the only check available on a video that has no labels.
    for tag, (cx, cy) in (("dlc", ("xd", "yd")), ("blob", ("xb", "yb"))):
        xo, yo = both[cx].to_numpy(), both[cy].to_numpy()
        mo = np.isfinite(xo) & np.isfinite(yo)
        if mo.sum() < 10:
            continue
        xe = np.arange(xo[mo].min(), xo[mo].max() + OCC_BIN, OCC_BIN)
        ye = np.arange(yo[mo].min(), yo[mo].max() + OCC_BIN, OCC_BIN)
        if len(xe) < 2 or len(ye) < 2:
            continue
        hh = np.histogram2d(xo[mo], yo[mo], bins=[xe, ye])[0]
        cov[f"{tag}_top_bin_frac"] = float(hh.max() / hh.sum())
        cov[f"{tag}_bins_occupied"] = int((hh > 0).sum())

    summary = {"bodypart": a.bodypart, "pcutoff": a.pcutoff,
               "per_split": rows, "coverage": cov}
    J.to_csv(f"{a.out}_errors.csv")
    with open(f"{a.out}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    for r in rows:
        if r.get("n"):
            print(f"  {r['split']:>7} {r['method']:>4}: n={r['n']:4d}  "
                  f"median {r['median_px']:5.2f} px  rmse {r['rmse_px']:5.2f}  "
                  f"debiased median {r['median_debiased_px']:5.2f}  "
                  f"bias ({r['bias_x_px']:+.1f},{r['bias_y_px']:+.1f})")
        else:
            print(f"  {r['split']:>7} {r['method']:>4}: no frames with a position")
    print(f"coverage over {cov['n_frames']} frames: DLC {100*cov['dlc_frac']:.1f}%, "
          f"blob {100*cov['blob_frac']:.1f}%, either {100*cov['either_frac']:.1f}%")
    if "dlc_top_bin_frac" in cov:
        print(f"DLC occupancy: {100*cov['dlc_top_bin_frac']:.1f}% of frames in the "
              f"densest {OCC_BIN}-px bin, {cov['dlc_bins_occupied']} bins visited "
              f"(a tracker stuck on hardware sits near 100% in one bin)")

    # ---- figure: which panels exist depends on which inputs were given
    has_lab = bool(np.isfinite(J.xh).any()) if len(J) else False
    has_blob = bool(np.isfinite(both.xb).any())
    n_panel = 2 * has_lab + 2 + (1 if ok.any() else 0)
    nrow = int(np.ceil(n_panel / 2))
    fig, axs = plt.subplots(nrow, 2, figsize=(11, 4.3 * nrow),
                            constrained_layout=True, squeeze=False)
    axs = axs.ravel()
    k = 0

    if has_lab:
        pri = "test" if (J.split == "test").any() else J.split.iloc[0]
        m = J.split == pri
        ax = axs[k]; k += 1
        for meth, col, c in (("DLC", "dlc_err", "#1f77b4"),
                             ("blob", "blob_err", "#d62728")):
            v = np.sort(J[col][m].dropna().to_numpy())
            if len(v):
                ax.plot(v, np.linspace(0, 100, len(v)), color=c, lw=1.8,
                        label=f"{meth} (n={len(v)})")
        ax.set(xlabel="error vs human label (px)", ylabel="% of frames below",
               title=f"Accuracy on {pri} frames")
        ax.legend(frameon=False)

        ax = axs[k]; k += 1
        for meth, (cx, cy), c in (("DLC", ("xd", "yd"), "#1f77b4"),
                                  ("blob", ("xb", "yb"), "#d62728")):
            ax.scatter(J[cx][m] - J.xh[m], J[cy][m] - J.yh[m], s=12, alpha=.5,
                       color=c, label=meth)
        ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
        ax.set(xlabel="x offset (px)", ylabel="y offset (px)",
               title="Offset from the human label\n"
                     "(a tight cloud off-centre is bias, not error)")
        ax.legend(frameon=False)

    # a short window makes coverage a sawtooth that reads as noise; average over
    # ~1% of the session so sustained dropouts are what shows
    w = max(100, len(both) // 100)
    rollc = pd.DataFrame({"DLC": np.isfinite(both.xd).astype(float),
                          "blob": np.isfinite(both.xb).astype(float)}
                         ).rolling(w, min_periods=1).mean()
    ax = axs[k]; k += 1
    ax.plot(both.index / 1000, 100 * rollc.DLC, color="#1f77b4", lw=1, label="DLC")
    if has_blob:
        ax.plot(both.index / 1000, 100 * rollc.blob, color="#d62728", lw=1, label="blob")
    ax.set(xlabel="frame (thousands)", ylabel="% of frames with a position",
           ylim=(-3, 103), title=f"Coverage across the session ({w}-frame mean)")
    ax.legend(frameon=False, ncol=2)

    # occupancy: the check that exposed the SuperAnimal failure, where a
    # confident-looking tracker had locked onto fixed rig hardware all session
    ax = axs[k]; k += 1
    xo, yo = both.xd.to_numpy(), both.yd.to_numpy()
    mo = np.isfinite(xo) & np.isfinite(yo)
    if "dlc_top_bin_frac" in cov:
        xe = np.arange(xo[mo].min(), xo[mo].max() + OCC_BIN, OCC_BIN)
        ye = np.arange(yo[mo].min(), yo[mo].max() + OCC_BIN, OCC_BIN)
        h = ax.hist2d(xo[mo], yo[mo], bins=[xe, ye], cmap="magma",
                      norm=LogNorm())[3]
        fig.colorbar(h, ax=ax, label="frames per bin")
        ax.invert_yaxis()
        ax.set(xlabel="x (px)", ylabel="y (px)", aspect="equal",
               title="Where DLC put the animal\n"
                     f"{100 * cov['dlc_top_bin_frac']:.1f}% of frames in one "
                     f"{OCC_BIN}-px bin, {cov['dlc_bins_occupied']} bins visited")

    if ok.any():
        ax = axs[k]; k += 1
        v = np.sort(agree[ok].to_numpy())
        ax.plot(v, np.linspace(0, 100, len(v)), color="#555", lw=1.8,
                label=f"raw, median {np.median(v):.1f} px")
        # the two methods measure different points on the animal, so a constant
        # offset is calibration; the residual scatter is the real disagreement
        vc = np.sort(resid.to_numpy())
        ax.plot(vc, np.linspace(0, 100, len(vc)), color="#2ca02c", lw=1.8,
                ls="--", label=f"offset removed, median {np.median(vc):.1f} px")
        ax.legend(frameon=False, loc="lower right")
        ax.set(xlabel="DLC-to-blob distance (px)", ylabel="% of frames below",
               title="Where the two methods agree\n"
                     f"({100 * cov['both_frac']:.0f}% of frames have both)")

    for ax in axs[k:]:
        ax.remove()
    fig.savefig(f"{a.out}_benchmark.png", dpi=150)
    print(f"wrote {a.out}_errors.csv, {a.out}_summary.json, {a.out}_benchmark.png")


if __name__ == "__main__":
    main()
