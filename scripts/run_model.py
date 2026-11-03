import json
import os
import resource
import sys
import time

import numpy as np

try:
    from .common import evaluate, load_data, log, out_dir
    from .models import MODELS
except ImportError:
    from common import evaluate, load_data, log, out_dir
    from models import MODELS

fast = "--fast" in sys.argv
names = [n for arg in sys.argv[1:] if not arg.startswith("--") for n in arg.split(",") if n]
if not names:
    sys.exit("usage: run_model.py <name> [<name> ...] [--fast]")
unknown = [n for n in names if n not in MODELS]
if unknown:  # fail before loading any data
    sys.exit(f"unknown model(s): {unknown}\navailable: {list(MODELS)}")

root = out_dir(fast)
D = load_data(fast)
log(f"data loaded: train {D['X_train'].shape}, val {D['X_val'].shape}, test {D['X_test'].shape}")


def run(name):
    family, kind, fn = MODELS[name]
    log(f"=== {name} ({family}, {kind}){' [FAST]' if fast else ''}")
    res = fn(D)

    out = {"name": name, "family": family, "task": kind, "fast": fast,
           "params": res["params"], "fit_time_s": res["fit_time_s"], "predict_time_s": res["predict_time_s"]}
    for split in ("val", "test"):
        reg, cls = res[split]
        out[split] = evaluate(D[f"y_{split}"], reg, D[f"uid_{split}"], cls)
        np.save(root / "preds" / f"{name}_{split}.npy", np.clip(reg, 1, 5).astype(np.float32))

    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    tmp = root / f"{name}.json.tmp"
    tmp.write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, root / f"{name}.json")
    log(f"DONE {name}: val rmse {out['val']['rmse']:.4f}  mae {out['val']['mae']:.4f}  ndcg@10 {out['val']['ndcg@10']:.4f}")


failed = []
for name in names:
    if (root / f"{name}.json").exists():
        log(f"skip {name} (already done; delete {root / (name + '.json')} to re-run)")
        continue
    try:
        run(name)
    except Exception as e:  # one failing model must not lose the others
        import traceback
        traceback.print_exc()
        log(f"FAILED {name}: {e!r}")
        failed.append(name)

log(f"finished. failed: {failed or 'none'}")
sys.exit(1 if failed else 0)