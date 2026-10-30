"""Track A (station level): features, splits and normalization for the 266-station demand/supply series.

Layout (docs/ASSUMPTIONS.md A-25): per time step a node's feature is [pick-ups of its k vicinity nodes (k),
drop-offs of the same nodes (k)] = 2k values (k = 9 -> 18); a sample stacks P steps time-major -> P*2k (= 216).
Splits (A-28): a sample belongs to the split in which its TARGET window lies entirely; inputs may reach back
into the previous split (they are past observations).  Normalization statistics come from the TRAIN steps only (A-27).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def vicinity_indices(sim: np.ndarray, k: int = 9) -> np.ndarray:
    """[N, k]: for every node the k nodes with the LARGEST weights in its row of `sim` (authors' rule
    `(-arr).argsort()[:9]`); ties -> lower index first (deterministic).  The node itself comes first when its weight is largest."""
    sim = np.asarray(sim, float)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("sim must be a square matrix")
    if not 1 <= k <= sim.shape[0]:
        raise ValueError("1 <= k <= N required")
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def step_features(pick: np.ndarray, drop: np.ndarray, vic: np.ndarray) -> np.ndarray:
    """[T, N] pick/drop + vic [N, k] -> F [T, N, 2k] = [pick of vicinity | drop of vicinity]."""
    pick, drop = np.asarray(pick), np.asarray(drop)
    if pick.shape != drop.shape or pick.shape[1] != vic.shape[0]:
        raise ValueError("pick/drop must be [T, N] and match vic [N, k]")
    return np.concatenate([pick[:, vic], drop[:, vic]], axis=2)


def window(F: np.ndarray, t: int, P: int, Q: int):
    """Sample whose target starts at step t: x = steps [t-P, t), y = steps [t, t+Q); both [N, steps*2k], time-major."""
    if t - P < 0 or t + Q > F.shape[0]:
        raise IndexError("window outside the series")
    N = F.shape[1]
    x = F[t - P:t].transpose(1, 0, 2).reshape(N, -1)
    y = F[t:t + Q].transpose(1, 0, 2).reshape(N, -1)
    return x, y


def split_bounds(T: int, val_steps: int, test_steps: int) -> dict:
    """Half-open step ranges. CCRNN setup: last 4 weeks = val (2 wks) + test (2 wks)."""
    if val_steps < 1 or test_steps < 1 or val_steps + test_steps >= T:
        raise ValueError("invalid split sizes")
    a, b = T - val_steps - test_steps, T - test_steps
    return dict(train=(0, a), val=(a, b), test=(b, T))


def sample_starts(rng_range: tuple, P: int, Q: int) -> np.ndarray:
    """Target-start steps t whose target [t, t+Q) lies inside [lo, hi) and whose input [t-P, t) exists."""
    lo, hi = rng_range
    return np.arange(max(lo, P), hi - Q + 1)


@dataclass(frozen=True)
class Stats:
    mean_pick: float
    std_pick: float
    mean_drop: float
    std_drop: float


def fit_stats(pick: np.ndarray, drop: np.ndarray, train_range: tuple) -> Stats:
    lo, hi = train_range
    p, d = np.asarray(pick[lo:hi], float), np.asarray(drop[lo:hi], float)
    if p.std() == 0 or d.std() == 0:
        raise ValueError("constant training series")
    return Stats(float(p.mean()), float(p.std()), float(d.mean()), float(d.std()))


def _apply(A: np.ndarray, st: Stats, k: int, inverse: bool) -> np.ndarray:
    A = np.asarray(A, float)
    lastdim = A.shape[-1]
    if lastdim % (2 * k) != 0:
        raise ValueError("last dimension must be a multiple of 2k")
    B = A.reshape(*A.shape[:-1], lastdim // (2 * k), 2, k).copy()
    for c, (m, s) in enumerate(((st.mean_pick, st.std_pick), (st.mean_drop, st.std_drop))):
        B[..., c, :] = B[..., c, :] * s + m if inverse else (B[..., c, :] - m) / s
    return B.reshape(A.shape)


def normalize(A, st: Stats, k: int = 9):
    return _apply(A, st, k, inverse=False)


def denormalize(A, st: Stats, k: int = 9):
    return _apply(A, st, k, inverse=True)
