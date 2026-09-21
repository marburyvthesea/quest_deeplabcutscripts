#!/usr/bin/env python
"""One animal, end to end: door detection -> behaviour/miniscope alignment.

Chains the three stages that were derived and validated on animal C:

  1. tmaze_geometry_transfer.py  measure the camera shift against C and carry
                                 the DS0 ROI + learned rail mask onto this
                                 animal's video
  2. door_state.py               per-frame door open/closed from the video
  3. align_tmaze.py              fit the behaviour camera's frame rate and each
                                 capture's offset, emit the alignment key and
                                 the traces joined to it

Expects the organised layout, one directory per animal:

  <pipeline-root>/<animal>/behavior/<protocol>.csv
                          /frame_timestamps/HH_MM_SS__*/timeStamps.csv
                          /saleae/<animal>_session*.csv
                          /traces/<animal>_Ca_traces_filtered_origHz.csv

Only the four-capture animals (C, F, K) are supported. m326 and m388 have a
single flat timeStamps.csv and one raw_*.csv instead of per-capture folders;
align_tmaze.py's rate fit needs several captures to cross-check its offsets, so
this driver refuses them rather than producing an unvalidated answer.

  python run_tmaze_pipeline.py --animal F \
      --pipeline-root /scratch/jma819/T_maze_recordings/dataOrganizedForPipeline \
      --video-dir     /scratch/jma819/T_maze_recordings/behaviorVIdeos \
      --tracking-dir  /scratch/jma819/T_maze_recordings/behaviorVIdeos/dlc_out_finetuned \
      --ref-animal C --outdir /scratch/jma819/T_maze_recordings/pipeline_out
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = {"m326", "m388"}


def run(cmd, tag):
    print(f"\n----- {tag}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd])
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"{tag} failed (exit {r.returncode}) after {dt:.0f}s")
    print(f"----- {tag} ok ({dt:.0f}s)", flush=True)
    return dt


def check_session(root, animal):
    if animal in LEGACY:
        raise SystemExit(
            f"{animal} uses the older single-capture layout (flat timeStamps.csv, "
            "raw_*.csv). align_tmaze.py validates its fit by requiring the "
            "per-capture offsets to agree, which needs several captures. Run "
            "that animal separately once a single-capture path is written.")
    need = ["behavior", "frame_timestamps", "saleae", "traces"]
    missing = [d for d in need if not os.path.isdir(f"{root}/{d}")]
    if missing:
        raise SystemExit(f"{root}: missing {missing}")
    caps = sorted(d for d in os.listdir(f"{root}/frame_timestamps")
                  if os.path.isdir(f"{root}/frame_timestamps/{d}"))
    sal = sorted(glob.glob(f"{root}/saleae/*.csv"))
    if len(caps) != len(sal) or not caps:
        raise SystemExit(f"{root}: {len(caps)} capture folders vs "
                         f"{len(sal)} saleae exports -- cannot pair")
    print(f"[{animal}] {len(caps)} captures: {', '.join(caps)}")
    return caps


def resolve_one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit(f"no {what} matching {pattern}")
    if len(hits) > 1:
        print(f"  WARNING {len(hits)} {what} match {pattern}; using newest")
        hits.sort(key=os.path.getmtime)
    return hits[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", required=True)
    p.add_argument("--pipeline-root", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--tracking-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--door", default="DS0")
    p.add_argument("--ref-animal", default="C")
    p.add_argument("--ref-geom", default=os.path.join(HERE, "maze_geometry_C.json"))
    p.add_argument("--ref-mask", default=os.path.join(HERE, "DS0_rail_mask.json"))
    p.add_argument("--force", action="store_true",
                   help="redo stages whose outputs already exist")
    a = p.parse_args()

    animal = a.animal
    root = os.path.join(a.pipeline_root, animal)
    out = os.path.join(a.outdir, animal)
    os.makedirs(out, exist_ok=True)
    caps = check_session(root, animal)

    video = resolve_one(f"{a.video_dir}/{animal}.avi", "video")
    ref_video = resolve_one(f"{a.video_dir}/{a.ref_animal}.avi", "reference video")
    tracking = resolve_one(f"{a.tracking_dir}/{animal}DLC_*.h5", "tracking h5")
    print(f"[{animal}] video    {video}")
    print(f"[{animal}] tracking {os.path.basename(tracking)}")
    for f in (a.ref_geom, a.ref_mask):
        if not os.path.isfile(f):
            raise SystemExit(f"reference file not found: {f}")

    summary = {"animal": animal, "session_root": root, "video": video,
               "tracking": tracking, "captures": caps, "outdir": out}
    timings = {}

    # ---- 1. geometry -------------------------------------------------------
    sys.path.insert(0, HERE)
    from tmaze_geometry_transfer import transfer
    geom = os.path.join(out, f"maze_geometry_{animal}.json")
    mask = os.path.join(out, f"{a.door}_rail_mask_{animal}.json")
    if a.force or not (os.path.isfile(geom) and os.path.isfile(mask)):
        t0 = time.time()
        geom, mask, dx, dy, resp = transfer(
            ref_video, a.ref_geom, video, animal, out,
            ref_mask_path=a.ref_mask, door=a.door)
        timings["geometry"] = time.time() - t0
        summary["camera_shift_px"] = {"dx": dx, "dy": dy, "response": resp}
    else:
        print(f"[{animal}] geometry already present; reusing (--force to redo)")
        summary["camera_shift_px"] = json.load(open(geom)).get("transfer")
    summary["geometry"], summary["mask"] = geom, mask

    # ---- 2. door state -----------------------------------------------------
    door_csv = os.path.join(out, f"{animal}_doorstate.csv")
    if a.force or not os.path.isfile(door_csv):
        timings["door_state"] = run(
            [sys.executable, os.path.join(HERE, "door_state.py"),
             "--video", video, "--geom", geom, "--door", a.door,
             "--mask", mask, "--out", door_csv,
             "--qc", os.path.join(out, f"{animal}_door_qc.png")],
            f"{animal}: door_state")
    else:
        print(f"[{animal}] {door_csv} exists; skipping (--force to redo)")
    summary["doorstate"] = door_csv

    # ---- 3. alignment ------------------------------------------------------
    prefix = os.path.join(out, animal)
    if a.force or not os.path.isfile(f"{prefix}_alignment_key.csv"):
        timings["align"] = run(
            [sys.executable, os.path.join(HERE, "align_tmaze.py"),
             "--session", root, "--doorstate", door_csv,
             "--tracking", tracking, "--out", prefix],
            f"{animal}: align_tmaze")
    else:
        print(f"[{animal}] {prefix}_alignment_key.csv exists; skipping")

    pj = f"{prefix}_alignment_params.json"
    if os.path.isfile(pj):
        summary["alignment"] = json.load(open(pj))
    summary["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    sp = os.path.join(out, f"{animal}_pipeline_summary.json")
    json.dump(summary, open(sp, "w"), indent=1)
    print(f"\n[{animal}] wrote {sp}")
    print(f"[{animal}] DONE")


if __name__ == "__main__":
    main()
