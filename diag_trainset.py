"""How many frames, from which animals, actually went into the fine-tune?

The axis diagnostic found the pose head puts body_center on the animal's
midline but at an arbitrary position along its length -- parts spaced 10-20 px
on a 38 px animal, with +-15 px of along-axis scatter, are unresolved. The
leading explanation is simply too few training frames: C, F and K carry 43, 37
and 22 labels between them, while H carries 286. If H was not in the split,
the fine-tune ran on ~100 frames and nearly three times that many are already
labelled and unused.

Reads DLC's own split pickle -- the authoritative record of what the shuffle
trained on -- and counts images per animal in train / test. Also counts the
labels present on disk, so "labelled but not used" is visible as the gap.
"""
import glob
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

P = "/projects/b1118/dlc_analysis/TmazeParts-jjm-2026-10-20"

docs = sorted(glob.glob(f"{P}/training-datasets/*/*/Documentation_data-*.pickle"))
if not docs:
    sys.exit(f"no Documentation_data-*.pickle under {P}/training-datasets")
print("split files found:")
for d in docs:
    print("  ", d.replace(P + "/", ""))

doc = docs[-1] if len(docs) == 1 else docs[-1]
print(f"\nreading {os.path.basename(doc)}")
with open(doc, "rb") as fh:
    obj = pickle.load(fh)
data, train_idx, test_idx = obj[0], list(obj[1]), list(obj[2])


def name_of(rec):
    """Image path of one formatted record, as a '/'-joined string.

    DLC stores it as the index TUPLE ('labeled-data', <animal>, 'img.png'),
    not a string -- str() on the tuple has no path separator, which is why
    the first pass put every image in '?'.
    """
    v = rec.get("image") if isinstance(rec, dict) else rec
    if isinstance(v, (list, tuple)):
        return "/".join(str(x) for x in v)
    return str(v or "")


def animal_of(n):
    """Second-to-last path component, whatever the layout.

    The first pass keyed on a literal 'labeled-data' component and every
    image fell through to '?', which hid the per-animal split entirely.
    DLC records image paths in several shapes across versions (relative,
    absolute, os.path.join of an index tuple), so take the parent directory
    rather than matching a fixed prefix.
    """
    parts = [p for p in re.split(r"[\\/]+", str(n)) if p]
    return parts[-2] if len(parts) >= 2 else "?"


names = [name_of(r) for r in data]
print(f"{len(names)} images in the shuffle; "
      f"{len(train_idx)} train / {len(test_idx)} test")

# The first run reported 276 images against 379+20=399 indices, which cannot
# both be true of one list -- so dump the structure rather than guess at it.
print("\n=== pickle structure")
print("  top-level:", type(obj).__name__,
      f"len={len(obj)}" if hasattr(obj, "__len__") else "")
for i, el in enumerate(obj if isinstance(obj, (list, tuple)) else []):
    print(f"   [{i}] {type(el).__name__:12s} "
          f"len={len(el) if hasattr(el, '__len__') else '-'}")
print("  sample image records:")
for r in data[:3]:
    print("   ", repr(r)[:200])
print("  sample parsed names:", [name_of(r) for r in data[:3]])
print(f"  index range: train {min(train_idx, default='-')}..",
      f"{max(train_idx, default='-')}, test {min(test_idx, default='-')}..",
      f"{max(test_idx, default='-')}")
n_oor = sum(1 for i in list(train_idx) + list(test_idx)
            if not (0 <= i < len(names)))
print(f"  indices outside 0..{len(names)-1}: {n_oor}")

# The split indices address the MERGED CollectedData dataframe that
# create_training_dataset wrote next to this pickle (399 rows), NOT the
# formatted `data` list (276 records). Indexing `data` with them is a
# category error: it silently drops every index >= len(data) -- here 123 of
# them -- and mis-attributes the rest. Load the merged frame and index that.
merged = sorted(glob.glob(os.path.join(os.path.dirname(doc),
                                       "CollectedData_*.h5")))
if not merged:
    sys.exit(f"no merged CollectedData_*.h5 beside {os.path.basename(doc)}; "
             "cannot map split indices to images")
Dm = pd.read_hdf(merged[0])
print(f"merged collected data: {len(Dm)} rows "
      f"({os.path.basename(merged[0])})")
m_animal = [t[-2] if isinstance(t, tuple) else animal_of(t) for t in Dm.index]
m_name = ["/".join(map(str, t)) if isinstance(t, tuple) else str(t)
          for t in Dm.index]

rows = []
for which, ids in (("train", train_idx), ("test", test_idx)):
    good = [int(i) for i in ids if 0 <= int(i) < len(Dm)]
    if len(good) != len(ids):
        print(f"  WARNING: {len(ids)-len(good)} {which} indices outside "
              f"0..{len(Dm)-1}")
    for a, k in Counter(m_animal[i] for i in good).items():
        rows.append({"animal": a, "split": which, "n": k})

# Which labelled frames never reached the formatted training records, and
# why -- a row with any NaN among the parts is the usual reason.
formatted = set(names)
missing = [n for n in m_name if n not in formatted]
print(f"\n{len(formatted)} of {len(Dm)} labelled frames reached the "
      f"formatted training records; {len(missing)} did not")
if missing:
    print("  dropped per animal:",
          dict(Counter(n.split("/")[-2] for n in missing)))
    nan_any = Dm.isna().any(axis=1).to_numpy()
    drop_mask = np.array([n not in formatted for n in m_name])
    print(f"  of the {drop_mask.sum()} dropped, "
          f"{int((nan_any & drop_mask).sum())} have at least one NaN label")
    print(f"  of the {(~drop_mask).sum()} kept, "
          f"{int((nan_any & ~drop_mask).sum())} have at least one NaN label")
T = (pd.DataFrame(rows).pivot_table(index="animal", columns="split",
                                    values="n", fill_value=0, aggfunc="sum")
     if rows else pd.DataFrame())

# what is labelled on disk, whether or not it was used
on_disk = {}
for d in sorted(glob.glob(f"{P}/labeled-data/*/CollectedData_*.h5")):
    a = os.path.basename(os.path.dirname(d))
    try:
        on_disk[a] = len(pd.read_hdf(d))
    except Exception as e:
        on_disk[a] = f"unreadable: {e}"

print("\n=== frames per animal")
T = T.reindex(sorted(set(list(T.index) + list(on_disk)))).fillna(0)
T["in_shuffle"] = T.sum(axis=1).astype(int)
T["labelled_on_disk"] = [on_disk.get(a, 0) for a in T.index]
T["unused"] = T.labelled_on_disk - T.in_shuffle
print(T.to_string())
print(f"\ntotal trained on: {len(train_idx)}   total labelled on disk: "
      f"{sum(v for v in on_disk.values() if isinstance(v, int))}")
T.to_csv("trainset_census.csv")
