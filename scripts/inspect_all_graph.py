"""Inspect all_graph.h5 (and adj_mx_taxi.pkl) and test what each matrix means.

    python scripts/inspect_all_graph.py --graph data/raw/all_graph.h5 --data data/raw/taxi_data.h5 --pkl data/raw/adj_mx_taxi.pkl
Prints shapes/statistics and, for every 266x266 matrix, its correlation with the Pearson-correlation matrix of the
taxi_pick and taxi_drop series (evidence for what 'pcc_tt' really is).  Writes results/graph_inspect.json.
"""
import argparse, json, os, pickle, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.graph_inspect import describe_matrix, pearson_matrix, offdiag_corr, direction_check

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="data/raw/all_graph.h5")
ap.add_argument("--data", default="data/raw/taxi_data.h5")
ap.add_argument("--pkl", default=None)
a = ap.parse_args()

series = {}
if a.data and os.path.exists(a.data):
    with h5py.File(a.data, "r") as hf:
        print("keys in data file:", list(hf.keys()))
        for k in hf.keys():
            series[k] = hf[k][:]
            print(f"  {k}: shape={series[k].shape} dtype={series[k].dtype} sum={float(series[k].sum()):.0f}")
refs = {k: pearson_matrix(v) for k, v in series.items() if v.ndim == 2}

rep = {}
mats = {}
with h5py.File(a.graph, "r") as hf:
    hf.visititems(lambda n, o: mats.__setitem__(n, o[:]) if isinstance(o, h5py.Dataset) else None)
if a.pkl:
    adj = pickle.load(open(a.pkl, "rb"), encoding="latin1")
    mats["pkl:adj[2]"] = np.asarray(adj[2])
for name, M in sorted(mats.items()):
    if M.ndim != 2:
        print(f"\n{name}: shape={M.shape} (not a matrix)"); continue
    d = describe_matrix(M)
    print(f"\n{name}: {d}")
    rep[name] = d
    if d["square"]:
        if "pick" in " ".join(series) and d["integer_valued"] and M.shape[0] == series.get("taxi_pick", np.zeros((1, 0))).shape[1]:
            dc = direction_check(M, series["taxi_pick"], series["taxi_drop"])
            rep[name]["direction_check"] = dc
            print("   direction check (is M[i,j] = trips i->j?):", {k: round(v, 4) for k, v in dc.items()})
        for rk, R in refs.items():
            if R.shape == M.shape:
                r = offdiag_corr(M, R)
                rep[name][f"corr_with_pearson_of_{rk}"] = r
                print(f"   corr(off-diagonal) with Pearson matrix of {rk}: {r:.4f}")
os.makedirs("results", exist_ok=True)
json.dump(rep, open("results/graph_inspect.json", "w"), indent=1, default=str)
print("\nwrote results/graph_inspect.json")
