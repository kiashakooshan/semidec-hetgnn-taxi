"""2-way clustering criteria compared in Fig. 11: k-means (sklearn, in assignment.py), k-medoids and fuzzy c-means.
All deterministic (no random initialisation) so runs are reproducible.
k-medoids: Park & Jun (2009), 'A simple and fast algorithm for k-medoids clustering' (paper ref [60]).
FCM: Bezdek (2013 reprint), m = 2 (paper ref [61])."""
from __future__ import annotations
import numpy as np


def _pairwise(pts):
    d = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def kmedoids(pts: np.ndarray, k: int = 2, max_iter: int = 100):
    """Returns (labels, medoid_indices). Medoids are always actual data points."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < k:
        raise ValueError("need at least k points")
    D = _pairwise(pts)
    denom = D.sum(1)                                           # sum_l d(i,l)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom[:, None] > 0, D / denom[:, None], 0.0).sum(0)   # v_j = sum_i d(i,j)/sum_l d(i,l)
    medoids = np.argsort(v, kind="stable")[:k]                 # Park-Jun initialisation
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members):
                new[c] = members[np.argmin(D[np.ix_(members, members)].sum(1))]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return np.argmin(D[:, medoids], axis=1), medoids


def fcm(pts: np.ndarray, k: int = 2, m: float = 2.0, max_iter: int = 300, tol: float = 1e-8):
    """Fuzzy c-means. Returns (hard_labels, centers, U[n,k]). Deterministic farthest-point initialisation."""
    pts = np.asarray(pts, float)
    if k != 2:
        raise NotImplementedError("2-way partitioning only")
    c0 = pts[np.argmax(((pts - pts.mean(0)) ** 2).sum(1))]
    c1 = pts[np.argmax(((pts - c0) ** 2).sum(1))]
    centers = np.stack([c0, c1])
    for _ in range(max_iter):
        d = np.sqrt(((pts[:, None, :] - centers[None]) ** 2).sum(-1))
        U = np.zeros_like(d)
        zero = d == 0
        has0 = zero.any(1)
        with np.errstate(divide="ignore"):
            inv = d ** (-2.0 / (m - 1))
        U[~has0] = inv[~has0] / inv[~has0].sum(1, keepdims=True)
        U[has0] = zero[has0] / zero[has0].sum(1, keepdims=True)
        Um = U ** m
        new = (Um.T @ pts) / Um.sum(0)[:, None]
        if np.abs(new - centers).max() < tol:
            centers = new
            break
        centers = new
    return np.argmax(U, axis=1), centers, U
