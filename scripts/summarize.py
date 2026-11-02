""""
Default shows VALIDATION metrics only. add --test once, for the final report
"""

import json
import sys
from pathlib import Path

import pandas as pd

fast, show_test = "--fast" in sys.argv, "--test" in sys.argv
root = Path("results_fast" if fast else "results")
split = "test" if show_test else "val"
rows = []
for f in sorted(root.glob("*.json")):
    r = json.load(open(f))
    m = r[split]
    rows.append({"name": r["name"], "family": r["family"], "task": r["task"],
                 "rmse": m["rmse"], "mae": m["mae"], "ndcg@10": m["ndcg@10"],
                 "acc": m.get("accuracy"), "f1_macro": m.get("f1_macro"),
                 "fit_min": r["fit_time_s"] / 60, "pred_s": r["predict_time_s"],
                 "n_train": r["params"]["n_train"], "peak_gb": r["peak_rss_mb"] / 1024,
                 "best_cfg": json.dumps(r["params"]["selected"])})
if not rows:
    sys.exit("no results yet")
df = pd.DataFrame(rows).sort_values("rmse")
df.to_csv(root / f"summary_{split}.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)
print(f"\n=== {split.upper()} results ({len(df)} experiments), sorted by RMSE ===")
print(df.drop(columns="best_cfg").round(4).to_string(index=False))
failed = [p.stem for p in root.glob("*.FAILED")]
if failed:
    print("\nFAILED (see logs/<name>.log):", failed)