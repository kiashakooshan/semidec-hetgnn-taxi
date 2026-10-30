"""Compare a CCRNN-style file with our own TLC counts (per-slot totals).

    python scripts/compare_with_ccrnn.py --h5 data/raw/taxi_data.h5
"""
import argparse, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.compare import compare_totals

ap = argparse.ArgumentParser()
ap.add_argument("--h5", default="data/raw/taxi_data.h5")
ap.add_argument("--category", default="taxi")
ap.add_argument("--ours", default="data/processed/zone_counts_30min.npz")
ap.add_argument("--max-lag", type=int, default=4)
a = ap.parse_args()

with h5py.File(a.h5, "r") as hf:
    print("keys in file:", list(hf.keys()))
    theirs = {k: hf[f"{a.category}_{k}"][:] for k in ("pick", "drop")}
ours = np.load(a.ours)
print({k: v.shape for k, v in theirs.items()}, "| ours:", ours["pickup"].shape)
rep = {}
for name, th, ou in (("pickup", theirs["pick"], ours["pickup"]), ("dropoff", theirs["drop"], ours["dropoff"])):
    r = compare_totals(ou, th, a.max_lag)
    rep[name] = r
    print(f"\n[{name}] slots={r['n_slots']}  corr@lag0={r['corr_lag0']:.4f}  best_lag={r['best_lag']} (corr {r['best_corr']:.4f})"
          f"\n   share of our trips present in theirs: mean ratio={r['ratio_mean']:.3f}, cv={r['ratio_cv']:.3f}, total ratio={r['total_ratio']:.3f}")
os.makedirs("results", exist_ok=True)
json.dump(rep, open("results/compare_ccrnn.json", "w"), indent=1)
print("\nwrote results/compare_ccrnn.json")
