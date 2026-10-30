"""Describe adjacency-like matrices and test what they mean by comparing them with quantities we can compute.

The authors' own notebook asks 'what is meant by trans_tt, pcc_tt, and dis_tt?'.  The names suggest distance, Pearson
correlation and transition, but that is only a guess; `offdiag_corr` against a matrix we compute from the demand series
(Pearson correlation) gives evidence instead of a guess.
"""
from __future__ import annotations
import numpy as np


def describe_matrix(M) -> dict:
    M = np.asarray(M, float)
    if M.ndim != 2:
        raise ValueError("2-D matrix expected")
    out = dict(shape=list(M.shape), square=M.shape[0] == M.shape[1], min=float(M.min()), max=float(M.max()),
               integer_valued=bool(np.allclose(M, np.round(M))), binary=bool(set(np.unique(M)) <= {0.0, 1.0}))
    if out["square"]:
        off = M - np.diag(np.diag(M))
        out.update(symmetric=bool(np.allclose(M, M.T)), diag_nonzero_frac=float(np.mean(np.diag(M) != 0)),
                   diag_mean=float(np.diag(M).mean()), nnz_offdiag=int(np.count_nonzero(off)),
                   row_sum_min=float(M.sum(1).min()), row_sum_max=float(M.sum(1).max()))
    return out


def pearson_matrix(series_tn) -> np.ndarray:
    """[T, N] -> [N, N] Pearson correlation between node series; constant series give 0 (not NaN)."""
    s = np.asarray(series_tn, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(s.T)
    return np.nan_to_num(c, nan=0.0)


def offdiag_corr(M, R) -> float:
    """Pearson r between the off-diagonal entries of two equally sized square matrices."""
    M, R = np.asarray(M, float), np.asarray(R, float)
    if M.shape != R.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("need two equally sized square matrices")
    mask = ~np.eye(M.shape[0], dtype=bool)
    a, b = M[mask], R[mask]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def direction_check(M, pick_tn, drop_tn) -> dict:
    """Is M[i, j] a count of trips from i to j?  Then row sums ~ pick-ups of i and column sums ~ drop-offs of j.
    Reports the four correlations (rows/cols against pick/drop totals per node) and the total ratios."""
    M, p, d = np.asarray(M, float), np.asarray(pick_tn, float).sum(0), np.asarray(drop_tn, float).sum(0)
    if M.shape != (len(p), len(p)) or len(d) != len(p):
        raise ValueError("M must be [N,N] and pick/drop [T,N]")

    def r(a, b):
        return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    row, col = M.sum(1), M.sum(0)
    return dict(total=float(M.sum()), pick_total=float(p.sum()), drop_total=float(d.sum()),
                total_over_pick=float(M.sum() / p.sum()),
                corr_row_pick=r(row, p), corr_col_drop=r(col, d), corr_row_drop=r(row, d), corr_col_pick=r(col, p))
