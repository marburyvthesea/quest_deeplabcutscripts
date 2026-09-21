#!/usr/bin/env bash
# Behaviour video -> calcium-aligned parquet, for one animal.
#
#   ./align_animal.sh F /path/F.avi /path/FDLC...h5 /path/organised/F
#
# Reference geometry and rail mask come from C (same rig, same day); the
# transfer step measures the camera shift rather than assuming it is zero.
# Stops at the first failure -- every stage here has an acceptance test and
# none of them should be skipped.
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: $0 <animal> <raw-video.avi> <DLC-tracking.h5> <organised-session-root> [bodypart]" >&2
    echo "example session root: /scratch/jma819/T_maze_recordings/dataOrganizedForPipeline/F" >&2
    exit 2
fi

ANIMAL=$1
VIDEO=$2
TRACKING=$3
SESSION=$4
BODYPART=${5:-mouse_center}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REFVID=${REFVID:-$(dirname -- "$VIDEO")/C.avi}
REFGEOM=${REFGEOM:-$SCRIPT_DIR/maze_geometry_C.json}
REFMASK=${REFMASK:-$SCRIPT_DIR/DS0_rail_mask.json}
GEOMDIR=${GEOMDIR:-$PWD}
TMAZE=${TMAZE:-$SCRIPT_DIR}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python || true)}

# Quest's base Python does not include the image-processing dependencies. Use
# the known DLC environment when it is available, while still allowing an
# explicit PYTHON_BIN override for other systems.
if { [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c "import cv2" >/dev/null 2>&1; } && \
        [ -x /home/jma819/.conda/envs/dlc3-torch/bin/python ]; then
    PYTHON_BIN=/home/jma819/.conda/envs/dlc3-torch/bin/python
fi

for path in "$VIDEO" "$TRACKING" "$REFVID" "$REFGEOM" "$REFMASK" \
        "$TMAZE/tmaze_geometry_transfer.py" "$TMAZE/door_state.py" \
        "$TMAZE/align_tmaze.py"; do
    if [ ! -f "$path" ]; then
        echo "required file not found: $path" >&2
        exit 1
    fi
done

for directory in behavior frame_timestamps saleae traces; do
    if [ ! -d "$SESSION/$directory" ]; then
        echo "organised session is missing: $SESSION/$directory" >&2
        echo "Use dataOrganizedForPipeline/$ANIMAL, not the raw recording directory." >&2
        exit 1
    fi
done

if [ -z "$PYTHON_BIN" ] || ! "$PYTHON_BIN" - <<'PY'
import sys
from importlib.util import find_spec

required = ("cv2", "matplotlib", "numpy", "pandas", "pyarrow", "scipy", "tables")
missing = [name for name in required if find_spec(name) is None]
if missing:
    print("missing Python modules: " + ", ".join(missing), file=sys.stderr)
    print("Activate dlc3-torch or set PYTHON_BIN to a compatible Python.", file=sys.stderr)
    raise SystemExit(1)
PY
then
    exit 1
fi

mkdir -p "$GEOMDIR"
echo "Python: $PYTHON_BIN"
echo "Session: $SESSION"

echo "== 1/3  geometry + rail mask transfer from C"
"$PYTHON_BIN" "$TMAZE/tmaze_geometry_transfer.py" \
    --ref-video "$REFVID" --ref-geom "$REFGEOM" --ref-mask "$REFMASK" \
    --video "$VIDEO" --animal "$ANIMAL" --outdir "$GEOMDIR"
echo "   look at $GEOMDIR/${ANIMAL}_geometry_qc.png before trusting anything below"

echo "== 2/3  door state from the rails"
"$PYTHON_BIN" "$TMAZE/door_state.py" \
    --video "$VIDEO" \
    --geom  "$GEOMDIR/maze_geometry_${ANIMAL}.json" \
    --mask  "$GEOMDIR/DS0_rail_mask_${ANIMAL}.json" \
    --out   "$GEOMDIR/${ANIMAL}_doorstate.csv"

echo "== 3/3  alignment"
"$PYTHON_BIN" "$TMAZE/align_tmaze.py" \
    --session   "$SESSION" \
    --doorstate "$GEOMDIR/${ANIMAL}_doorstate.csv" \
    --tracking  "$TRACKING" \
    --bodypart  "$BODYPART" \
    --out       "$GEOMDIR/$ANIMAL"

echo
echo "wrote $GEOMDIR/${ANIMAL}_traces_aligned.parquet"
echo "CHECK $GEOMDIR/${ANIMAL}_alignment_summary.png and the fit/validation block in"
echo "$GEOMDIR/${ANIMAL}_alignment_params.json before using it. An alignment error does"
echo "not add noise downstream -- it displaces every frame after the fault."
