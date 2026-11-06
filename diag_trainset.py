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
    if isinstance(rec, dict):
        return rec.get("image") or rec.get("image_path") or ""
    if isinstance(rec, (list, tuple)) and rec:
        return str(rec[0])
    return str(rec)


def animal_of(n):
    # labeled-data/<animal>/imgNNNNNNN.png, on either path separator
    m = re.search(r"labeled-data[\\/]+([^\\/]+)[\\/]", str(n))
    return m.group(1) if m else "?"


names = [name_of(r) for r in data]
print(f"{len(names)} images in the shuffle; "
      f"{len(train_idx)} train / {len(test_idx)} test")

rows = []
for which, ids in (("train", train_idx), ("test", test_idx)):
    c = Counter(animal_of(names[i]) for i in ids if 0 <= i < len(names))
    for a, k in c.items():
        rows.append({"animal": a, "split": which, "n": k})
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
