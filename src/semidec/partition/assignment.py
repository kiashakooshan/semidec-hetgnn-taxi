"""Node -> CLN (cloudlet) assignment: fixed grid and the paper's adaptive scheme (Algorithm 2).

Interpretation choices that the paper leaves open are listed in docs/ASSUMPTIONS.md (A-07..A-09).
Important property (docs/PAPER_ISSUES.md, P7): k-means works on node POSITIONS only; it never
looks at edges, so it is a heuristic for reducing cross-CLN edges, not a min-cut solver.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from .clustering import kmedoids, fcm
from ..graph.neighborhood import Adj


def cross_edge_count(adj: Adj, assignment: np.ndarray, pair: tuple[int, int] | None = None) -> int:
    """Number of undirected edges whose endpoints lie in different CLNs.
    If `pair=(a,b)` is given, count only edges between CLN a and CLN b."""
    cnt = 0
    for u, nbrs in adj.items():
        for w in nbrs:
            if u < w and assignment[u] != assignment[w]:
                if pair is None or {int(assignment[u]), int(assignment[w])} == set(pair):
                    cnt += 1
    return cnt


def grid_assignment(positions: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Fixed (coverage-based / 'uniform') assignment: equal-size cells over the bounding box.
    Label = row * n_cols + col."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be (n, 2)")
    lo, hi = positions.min(0), positions.max(0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    cell = np.floor((positions - lo) / span * np.array([n_cols, n_rows])).astype(int)
    cell[:, 0] = np.clip(cell[:, 0], 0, n_cols - 1)
    cell[:, 1] = np.clip(cell[:, 1], 0, n_rows - 1)
    return cell[:, 1] * n_cols + cell[:, 0]


def boundary_nodes(adj: Adj, assignment: np.ndarray) -> set[int]:
    """Footnote 3 of the paper: a node is a boundary node if it is connected to a node of another CLN."""
    return {u for u, nbrs in adj.items() if any(assignment[w] != assignment[u] for w in nbrs)}


def shared_region(adj: Adj, assignment: np.ndarray, a: int, b: int, hops: int) -> set[int]:
    """V_b for the pair (a,b): nodes of CLN a or b within `hops` hops (through a/b nodes only)
    of an endpoint of an a-b cross edge.  hops=0 -> only the endpoints."""
    in_ab = lambda x: int(assignment[x]) in (a, b)
    seeds = {u for u in adj if in_ab(u) and any(in_ab(w) and assignment[w] != assignment[u] for w in adj[u])}
    dist = {s: 0 for s in seeds}
    q = deque(seeds)
    while q:
        u = q.popleft()
        if dist[u] == hops:
            continue
        for w in adj[u]:
            if in_ab(w) and w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return set(dist)


@dataclass(frozen=True)
class PairResult:
    assignment: np.ndarray
    region: frozenset
    cross_before: int   # a-b cross edges before
    cross_after: int    # a-b cross edges after


def _two_way_labels(pts: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(pts)
    if method == "kmedoids":
        return kmedoids(pts, 2)[0]
    if method == "fcm":
        return fcm(pts, 2)[0]
    raise ValueError("method must be 'kmeans', 'kmedoids' or 'fcm'")


def adaptive_reassign_pair(adj: Adj, positions: np.ndarray, assignment: np.ndarray,
                           a: int, b: int, hops: int = 1, seed: int = 0,
                           method: str = "kmeans") -> PairResult:
    """One (u,v) iteration of Algorithm 2: 2-means on the positions of the shared region V_b,
    then map the two clusters to CLN a / CLN b so that as few nodes as possible change owner
    (cluster identity is arbitrary in k-means).  Nodes outside V_b never move."""
    assignment = np.asarray(assignment).copy()
    before = cross_edge_count(adj, assignment, (a, b))
    region = sorted(shared_region(adj, assignment, a, b, hops))
    pts = np.asarray(positions, dtype=float)[region]
    if len(region) < 2 or len(np.unique(pts, axis=0)) < 2:
        return PairResult(assignment, frozenset(region), before, before)
    labels = _two_way_labels(pts, method, seed)
    cur = assignment[region]
    agree_01 = int(np.sum((labels == 0) & (cur == a)) + np.sum((labels == 1) & (cur == b)))
    zero_to, one_to = (a, b) if agree_01 >= len(region) - agree_01 else (b, a)
    assignment[region] = np.where(labels == 0, zero_to, one_to)
    return PairResult(assignment, frozenset(region), before, cross_edge_count(adj, assignment, (a, b)))


def adaptive_assignment_all_pairs(adj: Adj, positions: np.ndarray, assignment: np.ndarray,
                                  hops: int = 1, seed: int = 0, method: str = "kmeans") -> np.ndarray:
    """Algorithm 2 outer loops: visit every adjacent CLN pair once, in sorted order.
    (Order matters because pairs share nodes; fixed order keeps runs reproducible.)"""
    assignment = np.asarray(assignment).copy()
    pairs = sorted({tuple(sorted((int(assignment[u]), int(assignment[w]))))
                    for u, nb in adj.items() for w in nb if assignment[u] != assignment[w]})
    for a, b in pairs:
        assignment = adaptive_reassign_pair(adj, positions, assignment, a, b, hops, seed, method).assignment
    return assignment
