"""Algorithm 1 (semi-decentralized inference) as an exact, testable computation.

One GNN layer (Eq. 5-7 with the sum aggregator, see docs/ASSUMPTIONS.md A-17):
        H' = act( sum_r  A^_r  H  W_r + b_r ),     A^_r = D^-1/2 (A_r + I) D^-1/2   (what PyG GCNConv computes)

* centralized_forward : one server holds everything.
* partitioned_forward : every cloudlet holds ONLY its own nodes' embeddings plus a halo of neighbour embeddings
  that it receives from the adjacent cloudlets at the start of EVERY layer (Algorithm 1, lines 8-11).
  Locality is enforced: a cloudlet's operator is sliced to (own nodes) x (own + halo) and we assert that nothing
  outside that block is needed.  The result must equal the centralized result exactly (tests/test_partitioned.py).
* Messages are counted: per layer, one embedding per (node, other cloudlet that has an edge to it).
"""
from __future__ import annotations
import numpy as np


def gcn_norm(A: np.ndarray) -> np.ndarray:
    """D^-1/2 (A+I) D^-1/2 with (weighted) degrees of A+I -- PyG GCNConv default (add_self_loops, symmetric)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    A_hat = A.copy()
    np.fill_diagonal(A_hat, 1.0)
    deg = A_hat.sum(1)
    dinv = 1.0 / np.sqrt(deg)
    return dinv[:, None] * A_hat * dinv[None, :]


def _layer(ops, H_cols, W, b, act):
    out = sum(ops[r] @ H_cols @ W[r] + b[r] for r in ops)
    return act(out)


def centralized_forward(A_rel: dict, X, W: list, b: list, acts: list):
    """W[l][r]: (d_in, d_out); b[l][r]: (d_out,); acts[l]: callable."""
    ops = {r: gcn_norm(A) for r, A in A_rel.items()}
    H = np.asarray(X, float)
    for l in range(len(W)):
        H = _layer(ops, H, W[l], b[l], acts[l])
    return H


def prune_cross_edges(A_rel: dict, assignment: np.ndarray, allowed_pairs: set) -> dict:
    """Remove edges between cloudlets that are NOT adjacent (the SC3 degradation the paper mentions)."""
    assignment = np.asarray(assignment)
    same_or_allowed = np.array([[(assignment[i] == assignment[j]) or
                                 (tuple(sorted((int(assignment[i]), int(assignment[j])))) in allowed_pairs)
                                 for j in range(len(assignment))] for i in range(len(assignment))])
    return {r: np.where(same_or_allowed, A, 0.0) for r, A in A_rel.items()}


def partitioned_forward(A_rel: dict, assignment, X, W: list, b: list, acts: list):
    """Returns (H_final[N, d_out], log). log['messages_per_layer'] = embeddings exchanged between cloudlets per layer."""
    assignment = np.asarray(assignment)
    N = len(assignment)
    clouds = sorted(set(int(c) for c in assignment))
    own = {c: np.flatnonzero(assignment == c) for c in clouds}
    ops_global = {r: gcn_norm(A) for r, A in A_rel.items()}      # in a real system: degrees exchanged once (init cost)

    # halo = nodes of other cloudlets that own-nodes depend on (any relation)
    halo, local_ops = {}, {}
    for c in clouds:
        dep = np.zeros(N, bool)
        for A in A_rel.values():
            dep |= (A[own[c]] != 0).any(0)
        h = np.flatnonzero(dep & (assignment != c))
        halo[c] = h
        cols = np.concatenate([own[c], h])
        local_ops[c] = {}
        for r, Ah in ops_global.items():
            rows = Ah[own[c]]
            outside = np.setdiff1d(np.flatnonzero((rows != 0).any(0)), cols)
            assert outside.size == 0, "operator needs nodes outside own+halo"      # locality check
            local_ops[c][r] = rows[:, cols]

    state = {c: np.asarray(X, float)[own[c]] for c in clouds}   # each cloudlet only keeps its own embeddings
    msgs = []
    for l in range(len(W)):
        # --- exchange: every cloudlet receives its halo embeddings from the owners (Algorithm 1, lines 8-11)
        full = np.zeros((N, state[clouds[0]].shape[1]))
        for c in clouds:
            full[own[c]] = state[c]
        n_msg = 0
        new_state = {}
        for c in clouds:
            H_cols = np.vstack([state[c], full[halo[c]]])       # own + received halo (nothing else is read)
            n_msg += len(halo[c])
            new_state[c] = _layer(local_ops[c], H_cols, W[l], b[l], acts[l])
        state = new_state
        msgs.append(n_msg)
    sent = {c: 0 for c in clouds}
    for c in clouds:                                   # every halo node of c is sent by its owner
        for h in halo[c]:
            sent[int(assignment[h])] += 1
    d_out = state[clouds[0]].shape[1]
    out = np.zeros((N, d_out))
    for c in clouds:
        out[own[c]] = state[c]
    return out, dict(messages_per_layer=msgs, halo_sizes={c: int(len(halo[c])) for c in clouds},
                     sent_sizes=sent, own_sizes={c: int(len(own[c])) for c in clouds})
