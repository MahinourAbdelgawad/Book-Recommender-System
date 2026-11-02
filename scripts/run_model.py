#run one model
# python -m scripts.run_model <name>
# Writes results/<name>.json only if everything succeeded

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

name, fast = sys.argv[1], "--fast" in sys.argv
root = out_dir(fast)
family, kind, fn = MODELS[name]

log(f"=== {name} ({family}, {kind}){' [FAST]' if fast else ''}")
D = load_data(fast)
log(f"data loaded: train {D['X_train'].shape}, val {D['X_val'].shape}, test {D['X_test'].shape}")
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
log(f"DONE val rmse {out['val']['rmse']:.4f}  mae {out['val']['mae']:.4f}  ndcg@10 {out['val']['ndcg@10']:.4f}")