"""Server-free federated averaging between adjacent cloudlets (paper Sec. IV-F, after ref [64]).

One round = every client (cloudlet) averages its own model with the models of its ADJACENT clients
(weighted by their local data sizes) and uses the result to initialise the next local training.
Floating tensors are averaged; non-float buffers are taken from the client itself.
"""
from __future__ import annotations
import torch


def average_state_dicts(states: list, weights: list | None = None) -> dict:
    if not states:
        raise ValueError("no states")
    w = torch.tensor(weights if weights is not None else [1.0] * len(states), dtype=torch.float64)
    if (w < 0).any() or w.sum() == 0:
        raise ValueError("weights must be non-negative with positive sum")
    w = w / w.sum()
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(float(wi) * s[k].to(torch.float64) for wi, s in zip(w, states)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def server_free_round(states: dict, neighbors: dict, sizes: dict | None = None) -> dict:
    """states: client -> state_dict;  neighbors: client -> set of adjacent clients (symmetric)."""
    for c, nb in neighbors.items():
        for o in nb:
            if c not in neighbors.get(o, set()):
                raise ValueError(f"adjacency not symmetric: {c} -> {o}")
    new = {}
    for c in states:
        group = [c] + sorted(neighbors.get(c, set()))
        new[c] = average_state_dicts([states[g] for g in group], [sizes[g] for g in group] if sizes else None)
    return new
