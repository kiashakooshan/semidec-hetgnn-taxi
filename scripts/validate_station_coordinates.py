"""Decide whether the recovered station coordinates can be trusted.  Run on the full data:

    python scripts/validate_station_coordinates.py
Tests
 1. KERNEL: exp(-d^2) built from the coordinates vs the real dis_bb (bike; coordinates are REAL Citi Bike ones), dis_tt (taxi),
    dis_tb (taxi x bike): which pairs are non-zero in both, the implied distance error in metres, row-sum differences.
 2. GEOGRAPHY (independent of dis_*): is each station inside a TLC zone, and does its pick-up series correlate best with
    the series of the zone it lies in?  (chance level for the median rank would be ~130 of 263)
 3. RADIO: how many stations have a neighbour within 100 m (the ad-hoc range used for SC2 in the paper)?
"""
import argparse, json, os, sys
import h5py
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.station_coords import load_taxi_coords, pairwise_km, gaussian_kernel, compare_kernel, own_zone_rank
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--taxi", default="data/external/station_coordinates_taxi266.csv")
ap.add_argument("--bike", default="data/external/station_mapping_bike250.csv")
ap.add_argument("--graph", default="data/raw/all_graph.h5")
ap.add_argument("--data", default="data/raw/taxi_data.h5")
ap.add_argument("--zones", default="data/raw/taxi_zones.zip")
ap.add_argument("--counts", default="data/processed/zone_counts_30min.npz")
a = ap.parse_args()

taxi = load_taxi_coords(a.taxi)
bike = pd.read_csv(a.bike).sort_values("h5_column")
assert (bike.h5_column.values == np.arange(len(bike))).all(), "bike h5_column must be 0..N-1"
rep = {}
with h5py.File(a.graph, "r") as hf:
    G = {k: hf[k][:] for k in ("dis_bb", "dis_tt", "dis_tb")}
Dt, Db = pairwise_km(taxi.lat, taxi.lon), pairwise_km(bike.lat, bike.lon)
Dtb = pairwise_km(np.r_[taxi.lat, bike.lat], np.r_[taxi.lon, bike.lon])[:len(taxi), len(taxi):]
for name, D, diag in (("dis_bb", Db, True), ("dis_tt", Dt, True), ("dis_tb", Dtb, False)):
    r = compare_kernel(gaussian_kernel(D, 2.0, set_diag=diag), G[name])
    rep["kernel_" + name] = r
    print(f"\n[{name}] {r}")

# --- geography (independent of dis_*) ---
if os.path.exists(a.zones) and os.path.exists(a.counts) and os.path.exists(a.data):
    import geopandas as gpd
    from semidec.data.zones import load_zones, find_id_col
    z = load_zones(a.zones); idc = find_id_col(z.columns)
    pts = gpd.GeoDataFrame(taxi[["station_id"]], geometry=gpd.points_from_xy(taxi.lon, taxi.lat), crs=4326).to_crs(z.crs)
    j = gpd.sjoin(pts, z[[idc, "geometry"]], how="left", predicate="within").drop_duplicates("station_id")
    zid = j[idc].values
    inside = ~pd.isna(zid)
    zcol = np.where(inside, np.nan_to_num(zid, nan=0).astype(int) - 1, -1)
    with h5py.File(a.data, "r") as hf:
        st = hf["taxi_pick"][:].astype(float)
    zones_pick = np.load(a.counts)["pickup"].astype(float)
    ranks = own_zone_rank(st, zones_pick, zcol)
    rr = ranks[ranks > 0]
    rep["geography"] = dict(inside_a_zone=float(inside.mean()), n_scored=int(len(rr)), median_rank=float(np.median(rr)),
                            share_rank1=float((rr == 1).mean()), share_rank_le3=float((rr <= 3).mean()),
                            share_rank_le10=float((rr <= 10).mean()), chance_median_rank=float((zones_pick.shape[1] + 1) / 2))
    print("\n[geography]", rep["geography"])
    bad = taxi.station_id.values[(ranks > 10) | (ranks == 0)]
    print("stations to inspect (outside zones or own-zone rank > 10):", bad.tolist())
else:
    print("\n[geography] skipped (need taxi_zones.zip, zone_counts_30min.npz and taxi_data.h5)")

# --- radio range ---
for r_km in (0.1, 0.2, 0.5):
    np.fill_diagonal(Dt, np.inf)
    rep[f"stations_with_neighbour_within_{int(r_km*1000)}m"] = int((Dt.min(1) <= r_km).sum())
print("\n[radio]", {k: v for k, v in rep.items() if k.startswith("stations_with")}, "of", len(taxi))
print("anchors<=3:", taxi.loc[taxi.n_bike_anchors <= 3, ["station_id", "n_bike_anchors"]].values.tolist())
os.makedirs("results", exist_ok=True)
json.dump(rep, open("results/coords_validation.json", "w"), indent=1, default=float)
print("wrote results/coords_validation.json")
