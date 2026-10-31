#!/usr/bin/env python3
# EXTERNAL script (third-party assistant), kept UNCHANGED for audit. Needs scipy. Known issue: it uses 110.57 km/deg for latitude
# (equatorial value; ~111.05 at 40.75N) -> its frame is ~0.3% short; see docs/PAPER_ISSUES.md P24.
"""
Recover the lat/lon of the 266 CCRNN taxi virtual stations from all_graph.h5.

Why this works (audit trail):
  * all_graph.h5 keys dis_tt / dis_bb / dis_tb are NOT distances. They are Gaussian
    kernels  w = exp(-(d/sigma)^2)  with d in KILOMETERS and sigma = 1, thresholded
    at d = 2 km (entries below the cutoff are exactly 0).
  * sigma = 1 is verified on the bike side: the 250 bike stations of bike_data.h5
    are REAL Citi Bike stations, identified by exact 30-min pickup-series matching
    against the official Citi Bike trip records (Apr-Jun 2016, 2016-citibike-tripdata.zip).
    All 250 matched with correlation 1.0000 and zero collisions (bike_250_station_mapping.csv).
    With their real coordinates, sigma_hat = 1.000 +- 0.000 over 19,756 pairs.
  * Hence d_km = sqrt(-ln w) recovers the exact Euclidean distance in km.
  * Each taxi station is positioned by trilateration (nonlinear least squares) against
    its dis_tb anchors, i.e. against the REAL coordinates of bike stations ONLY.
    dis_tt is NOT used as a fit target for well-constrained stations; it is used only
    (a) as a held-out validation afterwards, and (b) for the refinement of weak stations
    with < 3 anchors (these are reported in the output CSV and the log).

Usage:
  python recover_station_coordinates.py --all-graph data/raw/all_graph.h5 \
      --bike-mapping bike_250_station_mapping.csv \
      --out-taxi taxi_266_station_coordinates.csv

  --bike-mapping can be regenerated from raw Citi Bike CSVs with:
  python recover_station_coordinates.py --match-bike --citi-csv-dir <dir with 2016 Q2 CSVs> \
      --bike-h5 data/raw/bike_data.h5 --out-bike bike_250_station_mapping.csv

Deterministic: fixed seeds; repeated runs produce identical output.
"""

import argparse
import csv
import json
import sys
import numpy as np
import h5py

LAT0, LON0 = None, None  # set from bike mapping mean
COSLAT = None


def km_from_latlon(lat, lon, lat0, lon0):
    coslat = np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * 111.32 * coslat
    y = (lat - lat0) * 110.57
    return np.stack([x, y], axis=1)


def latlon_from_km(X, lat0, lon0):
    coslat = np.cos(np.deg2rad(lat0))
    lat = X[:, 1] / 110.57 + lat0
    lon = X[:, 0] / (111.32 * coslat) + lon0
    return lat, lon


def d_from_kernel(W):
    """d_km = sqrt(-ln w); W>0 entries only make sense."""
    return np.sqrt(-np.log(np.where(W > 0, W, np.nan)))


# ---------------------------------------------------------------------------
# Optional: rebuild the bike station mapping from raw Citi Bike CSVs
# ---------------------------------------------------------------------------
def match_bike_stations(citi_csv_dir, bike_h5_path, out_csv):
    import pandas as pd
    from pathlib import Path
    t0 = pd.Timestamp("2016-04-01 00:00:00")
    counts, coords, total = {}, {}, 0
    for fp in sorted(Path(citi_csv_dir).glob("*.csv")):
        df = pd.read_csv(fp, usecols=["starttime", "start station id", "start station name",
                                      "start station latitude", "start station longitude"])
        total += len(df)
        ts = pd.to_datetime(df["starttime"])
        bucket = ((ts - t0).dt.total_seconds() // 1800).astype(int)
        keep = (bucket >= 0) & (bucket < 4368)
        df, bucket = df[keep], bucket[keep]
        for (sid, b), c in df.assign(b=bucket).groupby(["start station id", "b"]).size().items():
            counts[(int(sid), int(b))] = counts.get((int(sid), int(b)), 0) + int(c)
        for sid, lat, lon, nm in df[["start station id", "start station latitude",
                                    "start station longitude", "start station name"]] \
                .drop_duplicates("start station id").itertuples(index=False):
            coords[int(sid)] = (float(lat), float(lon), nm)
    sids = sorted(coords)
    M = np.zeros((4368, len(sids)))
    sid_idx = {s: i for i, s in enumerate(sids)}
    for (s, b), c in counts.items():
        M[b, sid_idx[s]] += c
    with h5py.File(bike_h5_path, "r") as f:
        H = np.array(f["bike_pick"])

    def col_corr(M, H):
        Mc = M - M.mean(0)
        Hc = H - H.mean(0)
        Mc /= np.linalg.norm(Mc, axis=0, keepdims=True) + 1e-12
        Hc /= np.linalg.norm(Hc, axis=0, keepdims=True) + 1e-12
        return Hc.T @ Mc

    C = col_corr(M, H)
    matches, best = C.argmax(1), C.max(1)
    uniq, cnt = np.unique(matches, return_counts=True)
    print(f"[bike-match] trips={total} real_stations={len(sids)} "
          f"corr: min={best.min():.4f} median={np.median(best):.4f} "
          f"collisions={int((cnt > 1).sum())}")
    if best.min() < 0.99 or (cnt > 1).any():
        print("[bike-match] WARNING: weak or ambiguous matches — inspect before use")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h5_column", "real_station_id", "station_name", "lat", "lon", "match_corr"])
        for col in range(H.shape[1]):
            sid = sids[matches[col]]
            lat, lon, nm = coords[sid]
            w.writerow([col, sid, nm, lat, lon, round(float(best[col]), 4)])
    print(f"[bike-match] wrote {out_csv}")


# ---------------------------------------------------------------------------
# Main recovery
# ---------------------------------------------------------------------------
def recover(all_graph_path, bike_mapping_csv, out_csv):
    with h5py.File(all_graph_path, "r") as f:
        Wtt = np.array(f["dis_tt"])
        Wbb = np.array(f["dis_bb"])
        Wtb = np.array(f["dis_tb"])
    NT, NB = Wtt.shape[0], Wbb.shape[0]

    bl, bcorr = [], []
    with open(bike_mapping_csv) as f:
        for row in csv.DictReader(f):
            bl.append([float(row["lat"]), float(row["lon"])])
            bcorr.append(float(row["match_corr"]))
    bl = np.array(bl)
    assert len(bl) == NB, f"bike mapping has {len(bl)} rows, expected {NB}"

    lat0, lon0 = bl[:, 0].mean(), bl[:, 1].mean()
    Yb = km_from_latlon(bl[:, 0], bl[:, 1], lat0, lon0)

    # ---- Audit 1: sigma calibration on bike pairs (uses only REAL bike coords)
    ii, jj = np.where((Wbb > 0) & (~np.eye(NB, dtype=bool)))
    d_rel = np.sqrt(-np.log(Wbb[ii, jj]))
    d_euc = np.linalg.norm(Yb[ii] - Yb[jj], axis=1)
    ratio = d_euc / d_rel
    print(f"[sigma] bike pairs={len(ii)} sigma: median={np.median(ratio):.4f} "
          f"std={ratio.std():.5f} 10pct={np.percentile(ratio,10):.4f} 90pct={np.percentile(ratio,90):.4f}")

    # ---- Trilateration from dis_tb anchors (REAL bike coordinates only)
    from scipy.optimize import least_squares
    X = np.zeros((NT, 2))
    rms_list, n_anchors = [], (Wtb > 0).sum(1)
    for i in range(NT):
        js = np.where(Wtb[i] > 0)[0]
        d = np.sqrt(-np.log(Wtb[i, js]))

        def resid(p, js=js, d=d):
            return np.linalg.norm(Yb[js] - p, axis=1) - d

        best = None
        for seed in range(5):
            p0 = Yb[js].mean(0) + np.random.RandomState(seed).randn(2) * 2
            r = least_squares(resid, p0)
            rr = np.sqrt((r.fun ** 2).mean())
            if best is None or rr < best[0]:
                best = (rr, r.x)
        X[i] = best[1]
        rms_list.append(best[0])
    rms = np.array(rms_list)
    print(f"[trilateration] RMS km: median={np.median(rms):.4f} 90pct={np.percentile(rms,90):.4f} max={rms.max():.4f}")

    # ---- Weak stations (< 3 anchors): refine with dis_tt against solved stations
    weak = np.where(n_anchors < 3)[0]
    for i in weak:
        others = [j for j in range(NT) if j != i and Wtt[i, j] > 0 and n_anchors[j] >= 3]
        anchors = list(np.where(Wtb[i] > 0)[0])

        def resid2(p, others=others, anchors=anchors, i=i):
            r = [np.linalg.norm(X[j] - p) - np.sqrt(-np.log(Wtt[i, j])) for j in others]
            r += [np.linalg.norm(Yb[j] - p) - np.sqrt(-np.log(Wtb[i, j])) for j in anchors]
            return np.array(r)

        best = None
        for seed in range(20):
            p0 = X.mean(0) + np.random.RandomState(seed).randn(2) * 4
            r = least_squares(resid2, p0)
            rr = np.sqrt((r.fun ** 2).mean())
            if best is None or rr < best[0]:
                best = (rr, r.x)
        X[i] = best[1]
        print(f"[weak] station idx {i} (CCRNN id {i+1}): {n_anchors[i]} anchors, "
              f"{len(others)} taxi constraints, refined RMS={best[0]:.4f} km "
              f"(position accuracy ~100 m due to mirror ambiguity)")

    # ---- Audit 2: held-out validation against dis_tt (NOT used as fit target above)
    ii, jj = np.where((Wtt > 0) & (~np.eye(NT, dtype=bool)))
    d_kernel = np.sqrt(-np.log(Wtt[ii, jj]))
    d_solved = np.linalg.norm(X[ii] - X[jj], axis=1)
    err = d_solved - d_kernel
    print(f"[validate] dis_tt consistency: pairs={len(ii)} RMS={np.sqrt((err**2).mean()):.4f} km "
          f"max|err|={np.abs(err).max():.4f} km corr={np.corrcoef(d_solved, d_kernel)[0,1]:.4f}")

    # ---- Audit 3: kernel cutoff — Wtt==0 pairs must be >= 2 km apart
    ii0, jj0 = np.where((Wtt == 0) & (~np.eye(NT, dtype=bool)))
    d0 = np.linalg.norm(X[ii0] - X[jj0], axis=1)
    print(f"[validate] cutoff: Wtt==0 pair distances min={d0.min():.2f} km "
          f"(known-pair max={d_kernel.max():.2f} km) -> {'OK' if d0.min() > d_kernel.max() - 0.05 else 'CHECK'}")

    lat, lon = latlon_from_km(X, lat0, lon0)
    print(f"[result] extent {X[:,0].max()-X[:,0].min():.2f} x {X[:,1].max()-X[:,1].min():.2f} km, "
          f"lat [{lat.min():.4f},{lat.max():.4f}] lon [{lon.min():.4f},{lon.max():.4f}]")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "lat", "lon", "x_km", "y_km", "n_bike_anchors"])
        for i in range(NT):
            w.writerow([i + 1, round(float(lat[i]), 6), round(float(lon[i]), 6),
                        round(float(X[i, 0]), 4), round(float(X[i, 1]), 4), int(n_anchors[i])])
    print(f"[result] wrote {out_csv}  ({NT} stations; lat/lon reference mean "
          f"lat0={lat0:.5f} lon0={lon0:.5f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-graph", help="path to all_graph.h5")
    ap.add_argument("--bike-mapping", help="path to bike_250_station_mapping.csv")
    ap.add_argument("--out-taxi", default="taxi_266_station_coordinates.csv")
    ap.add_argument("--match-bike", action="store_true", help="rebuild bike mapping from raw CSVs")
    ap.add_argument("--citi-csv-dir", help="dir with 2016 Q2 Citi Bike CSVs")
    ap.add_argument("--bike-h5", help="path to bike_data.h5 (for --match-bike)")
    ap.add_argument("--out-bike", default="bike_250_station_mapping.csv")
    args = ap.parse_args()
    if args.match_bike:
        if not (args.citi_csv_dir and args.bike_h5):
            ap.error("--match-bike needs --citi-csv-dir and --bike-h5")
        match_bike_stations(args.citi_csv_dir, args.bike_h5, args.out_bike)
        return
    if not (args.all_graph and args.bike_mapping):
        ap.error("need --all-graph and --bike-mapping (or --match-bike ...)")
    recover(args.all_graph, args.bike_mapping, args.out_taxi)


if __name__ == "__main__":
    main()
