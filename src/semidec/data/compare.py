"""Sanity comparison of two per-slot demand tensors that should describe the same trips.

Used to check a CCRNN-style file (taxi_pick: [T, 266 virtual stations]) against OUR counts from the raw TLC files
(pickup: [T, 265 zones]).  The two spatial partitions differ, so nodes are not compared; the per-slot TOTALS are.
  * high correlation at lag 0   -> same time axis / same slot length / same source
  * ratio of totals             -> share of trips kept (CCRNN keeps only an 8.42 x 14.45 km box)
  * best lag != 0               -> a time offset between the two pipelines
"""
from __future__ import annotations
import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_totals(ours_tz: np.ndarray, theirs_tn: np.ndarray, max_lag: int = 4) -> dict:
    """lag L >= 0 means 'theirs lags ours by L slots': theirs[t+L] ~ ours[t]."""
    ours_tz, theirs_tn = np.asarray(ours_tz, float), np.asarray(theirs_tn, float)
    if ours_tz.ndim != 2 or theirs_tn.ndim != 2:
        raise ValueError("both inputs must be [T, nodes]")
    if ours_tz.shape[0] != theirs_tn.shape[0]:
        raise ValueError(f"different number of slots: ours {ours_tz.shape[0]} vs theirs {theirs_tn.shape[0]}")
    a, b = ours_tz.sum(1), theirs_tn.sum(1)
    T = len(a)
    if max_lag >= T:
        raise ValueError("max_lag must be smaller than the number of slots")
    corr_by_lag = {}
    for L in range(-max_lag, max_lag + 1):
        x, y = (a[:T - L], b[L:]) if L >= 0 else (a[-L:], b[:T + L])
        corr_by_lag[L] = _pearson(x, y)
    best = max(corr_by_lag, key=lambda k: -np.inf if np.isnan(corr_by_lag[k]) else corr_by_lag[k])
    ok = a > 0
    ratio = b[ok] / a[ok]
    return dict(n_slots=T, corr_lag0=corr_by_lag[0], best_lag=int(best), best_corr=corr_by_lag[best],
                ratio_mean=float(ratio.mean()), ratio_cv=float(ratio.std() / ratio.mean()) if ratio.mean() else float("nan"),
                total_ratio=float(b.sum() / a.sum()), corr_by_lag={int(k): v for k, v in corr_by_lag.items()})
