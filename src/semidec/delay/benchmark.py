"""Measure the two compute times the delay model needs, on THIS machine, with the REAL model code.

Timing does not depend on feature values, so inputs are constant tensors of the real shape; the graph structure
(edge_index per relation) is the real one passed in.  Returns the median over `repeats` runs, in milliseconds.
"""
from __future__ import annotations
import time
import statistics


def _median_ms(fn, repeats: int, warmup: int) -> float:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(ts))


def measure_layer_time_ms(edge_index: dict, n_nodes: int, in_dim: int = 216, out_dim: int = 216,
                          repeats: int = 20, warmup: int = 3) -> float:
    """One hetGNN layer (3 GCNConv + sum), i.e. 't_layer' of the paper."""
    import torch
    from ..model.hetgnn_lstm import HetGCNLayer
    layer = HetGCNLayer(in_dim, out_dim).eval()
    x = torch.zeros(n_nodes, in_dim)
    with torch.no_grad():
        return _median_ms(lambda: layer(x, edge_index), repeats, warmup)


def measure_mp_time_ms(edge_index: dict, n_nodes: int, dim: int = 216, repeats: int = 20, warmup: int = 3) -> float:
    """Message passing BY COMPUTATION, Eq. (5): M_r = A_r X for the three relations (sparse matmul)."""
    import torch
    x = torch.zeros(n_nodes, dim)
    mats = [torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1]), (n_nodes, n_nodes)).coalesce()
            for ei in edge_index.values()]
    with torch.no_grad():
        return _median_ms(lambda: [torch.sparse.mm(m, x) for m in mats], repeats, warmup)
