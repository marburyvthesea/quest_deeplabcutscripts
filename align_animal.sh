#!/usr/bin/env bash
# Behaviour video -> calcium-aligned parquet, for one animal.
#
#   ./align_animal.sh F /path/F.avi /path/FDLC...h5 /path/organised/F_session
#
# Reference geometry and rail mask come from C (same rig, same day); the
# transfer step measures the camera shift rather than assuming it is zero.
# Run from the quest_deeplabcutscripts checkout. Stops at the first failure --
# every stage here has an acceptance test and none of them should be skipped.
set -euo pipefail

ANIMAL=${1:?animal letter}
VIDEO=${2:?raw behaviour video (NOT the DLC _labeled.mp4)}
TRACKING=${3:?DeepLabCut .h5}
SESSION=${4:?organised session root}
BODYPART=${5:-mouse_center}

REFVID=${REFVID:-C.avi}
REFGEOM=${REFGEOM:-maze_geometry_C.json}
REFMASK=${REFMASK:-DS0_rail_mask.json}
GEOMDIR=${GEOMDIR:-.}
TMAZE=${TMAZE:-$HOME/TMazeMiniscopeAnalysis}

echo "== 1/3  geometry + rail mask transfer from C"
python "$TMAZE/tmaze_geometry_transfer.py" \
    --ref-video "$REFVID" --ref-geom "$REFGEOM" --ref-mask "$REFMASK" \
    --video "$VIDEO" --animal "$ANIMAL" --outdir "$GEOMDIR"
echo "   look at ${ANIMAL}_geometry_qc.png before trusting anything below"

echo "== 2/3  door state from the rails"
python door_state.py \
    --video "$VIDEO" \
    --geom  "$GEOMDIR/maze_geometry_${ANIMAL}.json" \
    --mask  "$GEOMDIR/DS0_rail_mask_${ANIMAL}.json" \
    --out   "${ANIMAL}_doorstate.csv"

echo "== 3/3  alignment"
python align_tmaze.py \
    --session   "$SESSION" \
    --doorstate "${ANIMAL}_doorstate.csv" \
    --tracking  "$TRACKING" \
    --bodypart  "$BODYPART" \
    --out       "$ANIMAL"

echo
echo "wrote ${ANIMAL}_traces_aligned.parquet"
echo "CHECK ${ANIMAL}_alignment_summary.png and the fit/validation block in"
echo "${ANIMAL}_alignment_params.json before using it. An alignment error does"
echo "not add noise downstream -- it displaces every frame after the fault."
