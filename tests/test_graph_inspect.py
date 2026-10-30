import numpy as np, pytest
from semidec.data.graph_inspect import describe_matrix, pearson_matrix, offdiag_corr, direction_check

M = np.array([[1, .5, 0], [.5, 1, .2], [0, .2, 1]])

def test_describe_hand():
    d = describe_matrix(M)
    assert d["symmetric"] and d["square"] and d["diag_nonzero_frac"] == 1.0 and d["nnz_offdiag"] == 4
    assert (d["min"], d["max"]) == (0.0, 1.0) and not d["binary"] and not d["integer_valued"]
    assert d["row_sum_min"] == pytest.approx(1.2) and d["row_sum_max"] == pytest.approx(1.7)
    assert describe_matrix(np.eye(3))["binary"] and not describe_matrix(np.array([[0, 1], [0, 0.0]]))["symmetric"]

def test_pearson_matrix_hand():
    s = np.stack([[1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1]], 1).astype(float)     # [T=4, N=3]
    assert np.allclose(pearson_matrix(s), [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    s[:, 1] = 5.0                                                                   # constant series -> 0, not NaN
    assert np.isfinite(pearson_matrix(s)).all() and pearson_matrix(s)[0, 1] == 0.0

def test_offdiag_corr_ignores_diagonal():
    R = M.copy(); np.fill_diagonal(R, 99.0)
    assert offdiag_corr(M, R) == pytest.approx(1.0)
    flipped = 1.0 - M; np.fill_diagonal(flipped, 0.0)
    assert offdiag_corr(M, flipped) == pytest.approx(-1.0)
    assert np.isnan(offdiag_corr(M, np.ones((3, 3))))
    with pytest.raises(ValueError): offdiag_corr(M, np.ones((2, 2)))


def test_direction_check_hand():
    T_ = np.array([[1, 2, 3], [0, 1, 0], [2, 0, 4]], float)           # trips i -> j ; rows [6,1,6], cols [3,3,7]
    pick = np.array([[6.0, 1.0, 6.0]]); drop = np.array([[3.0, 3.0, 7.0]])   # T=1, N=3
    d = direction_check(T_, pick, drop)
    assert d["corr_row_pick"] == pytest.approx(1.0) and d["corr_col_drop"] == pytest.approx(1.0)
    assert d["total"] == 13 and d["pick_total"] == 13 and d["total_over_pick"] == pytest.approx(1.0)
    assert d["corr_row_drop"] < 0.99 and d["corr_col_pick"] < 0.99          # the other direction does NOT match
    with pytest.raises(ValueError): direction_check(np.ones((2, 2)), pick, drop)
