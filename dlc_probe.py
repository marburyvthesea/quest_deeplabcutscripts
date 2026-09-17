#!/usr/bin/env python
"""
Report what the installed DeepLabCut actually exposes. Runs in seconds on a
login node and needs no GPU.

The SuperAnimal fine-tuning entry points moved between DLC 2.3, 3.0 and later,
so run this before submitting dlc_finetune.py and paste the output back. An API
mismatch found here costs a minute; found on a GPU allocation it costs the
allocation.

  python dlc_probe.py
"""
import importlib
import inspect

import deeplabcut

print("deeplabcut", deeplabcut.__version__)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)

for fn in ("create_training_dataset", "train_network", "evaluate_network",
           "analyze_videos"):
    f = getattr(deeplabcut, fn, None)
    print(f"\n{fn}: ", end="")
    print(str(inspect.signature(f)) if f else "MISSING")

for mod, attr in [("deeplabcut.core.weight_init", "WeightInitialization"),
                  ("deeplabcut.pose_estimation_pytorch.config", None),
                  ("deeplabcut.modelzoo.utils", "create_conversion_table"),
                  ("deeplabcut.modelzoo.api", None)]:
    try:
        m = importlib.import_module(mod)
        if attr:
            o = getattr(m, attr, None)
            sig = ""
            if o is not None and hasattr(o, "build"):
                sig = " .build" + str(inspect.signature(o.build))
            print(f"\n{mod}.{attr}: {'present' if o else 'MISSING'}{sig}")
        else:
            print(f"\n{mod}: present; "
                  f"{[n for n in dir(m) if not n.startswith('_')][:14]}")
    except Exception as e:
        print(f"\n{mod}: import failed ({type(e).__name__})")

try:
    from deeplabcut.modelzoo import SuperAnimalOptions  # noqa: F401
    print("\nSuperAnimalOptions: present")
except Exception:
    pass
try:
    from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path
    print("\ndlc path:", get_deeplabcut_path())
except Exception as e:
    print("\ndlc path unavailable:", type(e).__name__)
