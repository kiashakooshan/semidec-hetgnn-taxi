"""Plain-Python graph helpers (undirected graphs as dict[int, set[int]])."""
from __future__ import annotations
from collections import deque
from typing import Iterable

Adj = dict[int, set[int]]


def build_adj(n_nodes: int, edges: Iterable[tuple[int, int]]) -> Adj:
    """Undirected simple graph on nodes 0..n_nodes-1. Self-loops/duplicates are dropped."""
    adj: Adj = {i: set() for i in range(n_nodes)}
    for u, v in edges:
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError(f"edge ({u},{v}) out of range 0..{n_nodes-1}")
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def degree(adj: Adj, i: int) -> int:
    return len(adj[i])


def hop_shells(adj: Adj, source: int, L: int) -> list[set[int]]:
    """shells[l-1] = set of nodes at *exactly* l hops from `source`, l = 1..L."""
    if source not in adj:
        raise ValueError(f"unknown node {source}")
    if L < 1:
        raise ValueError("L must be >= 1")
    dist = {source: 0}
    q = deque([source])
    shells: list[set[int]] = [set() for _ in range(L)]
    while q:
        u = q.popleft()
        if dist[u] == L:
            continue
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                shells[dist[w] - 1].add(w)
                q.append(w)
    return shells


def l_hop_ball(adj: Adj, source: int, L: int) -> set[int]:
    """All nodes within L hops of source (excluding source)."""
    return set().union(*hop_shells(adj, source, L))
