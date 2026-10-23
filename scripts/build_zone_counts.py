"""Build zone x 30-min pick-up / drop-off count tensors from the TLC parquet files.

    python scripts/build_zone_counts.py
    python scripts/build_zone_counts.py --slot-minutes 60

Output (data/processed/):
    zone_counts_<slot>min.npz   pickup[T,Z], dropoff[T,Z], zone_ids[Z], slot_start[T] (ISO strings)
    zone_counts_<slot>min.report.json   every excluded event is counted here
"""
import argparse, json, os, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.aggregate import SlotGrid, count_events

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--months", nargs="+", default=["2016-04", "2016-05", "2016-06"])
ap.add_argument("--start", default="2016-04-01T00:00:00")
ap.add_argument("--end", default="2016-07-01T00:00:00")
ap.add_argument("--slot-minutes", type=int, default=30)
ap.add_argument("--n-zones", type=int, default=265)
ap.add_argument("--out", default="data/processed")
a = ap.parse_args()

grid = SlotGrid(np.datetime64(a.start), np.datetime64(a.end), a.slot_minutes)
T, Z = grid.n_slots, a.n_zones
pickup = np.zeros((T, Z), np.int64); dropoff = np.zeros((T, Z), np.int64)
rep = dict(slot_minutes=a.slot_minutes, n_slots=T, n_zones=Z, window=[a.start, a.end], files={})
tot_rows = 0
for m in a.months:
    path = os.path.join(a.raw, f"yellow_tripdata_{m}.parquet")
    pf = pq.ParquetFile(path)
    fs = dict(rows=pf.metadata.num_rows, pu=dict(n_total=0, n_counted=0, n_outside_window=0, n_bad_zone_inside_window=0),
              do=dict(n_total=0, n_counted=0, n_outside_window=0, n_bad_zone_inside_window=0), dropoff_before_pickup=0)
    for b in pf.iter_batches(batch_size=1_000_000,
            columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]):
        pu_t = b.column("tpep_pickup_datetime").to_numpy(zero_copy_only=False)
        do_t = b.column("tpep_dropoff_datetime").to_numpy(zero_copy_only=False)
        c, s = count_events(pu_t, b.column("PULocationID").to_numpy(zero_copy_only=False), grid, Z)
        pickup += c; [fs["pu"].__setitem__(k, fs["pu"][k] + v) for k, v in s.items()]
        c, s = count_events(do_t, b.column("DOLocationID").to_numpy(zero_copy_only=False), grid, Z)
        dropoff += c; [fs["do"].__setitem__(k, fs["do"][k] + v) for k, v in s.items()]
        fs["dropoff_before_pickup"] += int((do_t < pu_t).sum())
    rep["files"][m] = fs; tot_rows += fs["rows"]
    print(m, fs)

rep["total_rows"] = tot_rows
rep["pickup_total_counted"] = int(pickup.sum()); rep["dropoff_total_counted"] = int(dropoff.sum())
rep["checks"] = dict(
    pickup_counts_plus_excluded_equals_rows=bool(sum(f["pu"]["n_counted"] + f["pu"]["n_outside_window"] + f["pu"]["n_bad_zone_inside_window"] for f in rep["files"].values()) == tot_rows),
    dropoff_counts_plus_excluded_equals_rows=bool(sum(f["do"]["n_counted"] + f["do"]["n_outside_window"] + f["do"]["n_bad_zone_inside_window"] for f in rep["files"].values()) == tot_rows),
    zones_with_pickups=int((pickup.sum(0) > 0).sum()), zones_with_dropoffs=int((dropoff.sum(0) > 0).sum()),
    pickup_per_slot_mean=float(pickup.sum(1).mean()), pickup_per_slot_min=int(pickup.sum(1).min()), pickup_per_slot_max=int(pickup.sum(1).max()))
os.makedirs(a.out, exist_ok=True)
stem = os.path.join(a.out, f"zone_counts_{a.slot_minutes}min")
np.savez_compressed(stem + ".npz", pickup=pickup, dropoff=dropoff, zone_ids=np.arange(1, Z + 1),
                    slot_start=grid.slot_starts().astype(str))
json.dump(rep, open(stem + ".report.json", "w"), indent=1)
print("\nCHECKS:", json.dumps(rep["checks"], indent=1)); print("wrote", stem + ".npz")
