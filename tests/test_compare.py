import numpy as np, pytest
from semidec.data.compare import compare_totals

A = np.array([1, 3, 2, 5, 4, 7, 6, 9], float)
OURS = np.stack([A * 0.6, A * 0.4], 1)                 # [T=8, Z=2]; row totals = A

def test_same_series_scaled():
    r = compare_totals(OURS, (0.5 * A)[:, None])        # theirs = half of ours, one node
    assert r["corr_lag0"] == pytest.approx(1.0) and r["best_lag"] == 0
    assert r["ratio_mean"] == pytest.approx(0.5) and r["ratio_cv"] == pytest.approx(0.0) and r["total_ratio"] == pytest.approx(0.5)

def test_detects_one_slot_delay():
    delayed = np.concatenate([[0.0], 0.5 * A[:-1]])      # theirs[t] = 0.5 * ours[t-1]  ->  theirs lags ours by 1
    r = compare_totals(OURS, delayed[:, None], max_lag=2)
    assert r["best_lag"] == 1 and r["best_corr"] == pytest.approx(1.0) and r["corr_lag0"] < 0.99

def test_validation():
    with pytest.raises(ValueError): compare_totals(OURS, np.ones((5, 1)))
    with pytest.raises(ValueError): compare_totals(A, A)
    with pytest.raises(ValueError): compare_totals(OURS, OURS, max_lag=8)
