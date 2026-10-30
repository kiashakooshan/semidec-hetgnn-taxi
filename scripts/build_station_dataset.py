"""Track A: build the station-level dataset from taxi_data.h5 + the similarity matrix.

    python scripts/build_station_dataset.py
Saves data/processed/station_dataset.npz with the normalized per-step features F [T, N, 18] (small), the vicinity
indices, split bounds and normalization statistics.  Windows are cut on the fly with semidec.data.stations.window().
"""
import argparse, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.stations import (vicinity_indices, step_features, split_bounds, sample_starts, fit_stats,
                                   normalize, window)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/taxi_data.h5")
ap.add_argument("--graph", default="data/raw/all_graph.h5")
ap.add_argument("--sim-key", default="dis_tt")
ap.add_argument("--P", type=int, default=12); ap.add_argument("--Q", type=int, default=12)
ap.add_argument("--k", type=int, default=9)
ap.add_argument("--val-steps", type=int, default=672); ap.add_argument("--test-steps", type=int, default=672)
ap.add_argument("--out", default="data/processed/station_dataset.npz")
a = ap.parse_args()

with h5py.File(a.data, "r") as hf:
    pick, drop = hf["taxi_pick"][:].astype(float), hf["taxi_drop"][:].astype(float)
with h5py.File(a.graph, "r") as hf:
    sim = hf[a.sim_key][:]
T, N = pick.shape
vic = vicinity_indices(sim, a.k)
assert (vic[:, 0] == np.arange(N)).all() or print("note: node itself is not always the first vicinity entry")
bounds = split_bounds(T, a.val_steps, a.test_steps)
st = fit_stats(pick, drop, bounds["train"])
F = normalize(step_features(pick, drop, vic), st, a.k).astype(np.float32)
rep = dict(T=T, N=N, k=a.k, P=a.P, Q=a.Q, bounds=bounds, stats=st.__dict__,
           samples={n: int(len(sample_starts(r, a.P, a.Q))) for n, r in bounds.items()},
           self_first_fraction=float((vic[:, 0] == np.arange(N)).mean()),
           F_shape=list(F.shape), F_abs_max=float(np.abs(F).max()))
x, y = window(F, bounds["test"][0], a.P, a.Q)
rep["window_shapes"] = [list(x.shape), list(y.shape)]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out, F=F, vic=vic, bounds=json.dumps(bounds), stats=json.dumps(st.__dict__))
json.dump(rep, open("results/station_dataset_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1)); print("wrote", a.out)
