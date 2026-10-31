"""Recovered coordinates of the 266 CCRNN taxi stations and the tests that decide whether to trust them.

Hypothesis (to be tested, not assumed): all_graph.h5's dis_* matrices are Gaussian kernels exp(-d^2) of the
Euclidean distance d in km, set to 0 beyond about 2 km, diagonal 1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088
TAXI_COLS = ["station_id", "lat", "lon", "x_km", "y_km", "n_bike_anchors"]
NYC_LAT, NYC_LON = (40.4, 41.0), (-74.3, -73.6)


def load_taxi_coords(path: str, n_expected: int | None = 266) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != TAXI_COLS:
        raise ValueError(f"columns must be {TAXI_COLS}, got {list(df.columns)}")
    if n_expected is not None and len(df) != n_expected:
        raise ValueError(f"expected {n_expected} rows, got {len(df)}")
    if df.isna().any().any():
        raise ValueError("NaN in coordinates")
    if not (df.station_id.values == np.arange(1, len(df) + 1)).all():
        raise ValueError("station_id must be 1..N in column order of taxi_data.h5")
    if not (df.lat.between(*NYC_LAT).all() and df.lon.between(*NYC_LON).all()):
        raise ValueError("coordinates outside the NYC box")
    if df.duplicated(["lat", "lon"]).any():
        raise ValueError("duplicate station positions")
    return df


def pairwise_km(lat, lon) -> np.ndarray:
    """Great-circle (haversine) distances in km."""
    p, l = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    a = np.sin((p[:, None] - p[None]) / 2) ** 2 + np.cos(p[:, None]) * np.cos(p[None]) * np.sin((l[:, None] - l[None]) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def gaussian_kernel(D: np.ndarray, cutoff_km: float = 2.0, set_diag: bool = True) -> np.ndarray:
    """exp(-d^2) for d < cutoff, 0 beyond; diagonal 1 for square self-similarity matrices (set_diag=False for taxi x bike)."""
    D = np.asarray(D, float)
    W = np.where(D < cutoff_km, np.exp(-D ** 2), 0.0)
    if set_diag:
        np.fill_diagonal(W, 1.0)
    return W


def compare_kernel(W_pred: np.ndarray, W_true: np.ndarray) -> dict:
    """How well does a kernel built from coordinates reproduce the file's matrix?"""
    if W_pred.shape != W_true.shape:
        raise ValueError("shape mismatch")
    off = ~np.eye(W_true.shape[0], dtype=bool) if W_true.shape[0] == W_true.shape[1] else np.ones(W_true.shape, bool)
    p, t = (W_pred > 0) & off, (W_true > 0) & off
    both = p & t
    d_pred, d_true = np.sqrt(-np.log(W_pred[both])), np.sqrt(-np.log(W_true[both]))
    err_m = np.abs(d_pred - d_true) * 1000.0
    rs_p, rs_t = W_pred.sum(1), W_true.sum(1)   # for rectangular matrices: sums over the other station type
    return dict(nnz_pred=int(p.sum()), nnz_true=int(t.sum()), nnz_common=int(both.sum()),
                only_pred=int((p & ~t).sum()), only_true=int((t & ~p).sum()),
                dist_err_m_median=float(np.median(err_m)) if err_m.size else float("nan"),
                dist_err_m_p95=float(np.percentile(err_m, 95)) if err_m.size else float("nan"),
                dist_err_m_max=float(err_m.max()) if err_m.size else float("nan"),
                rowsum_max_abs_diff=float(np.abs(rs_p - rs_t).max()))


def own_zone_rank(station_series: np.ndarray, zone_series: np.ndarray, zone_col_of_station: np.ndarray) -> np.ndarray:
    """For each station: rank (1 = best) of the zone it lies in, among all zones, by Pearson correlation of the
    time series.  station_series [T,S], zone_series [T,Z], zone_col_of_station [S] (-1 = not inside any zone -> rank 0)."""
    S = station_series.shape[1]
    ranks = np.zeros(S, int)
    zc = (zone_series - zone_series.mean(0)) / np.where(zone_series.std(0) == 0, 1, zone_series.std(0))
    for s in range(S):
        z = zone_col_of_station[s]
        if z < 0:
            continue
        x = station_series[:, s]
        x = (x - x.mean()) / (x.std() if x.std() else 1.0)
        r = zc.T @ x / len(x)
        ranks[s] = 1 + int((r > r[z]).sum())
    return ranks
