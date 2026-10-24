"""Algorithm 1 exactness.  Inputs are fixed deterministic arrays built from np.arange (hand-checkable), not random data."""
import numpy as np, pytest
from semidec.inference.partitioned import (gcn_norm, centralized_forward, partitioned_forward, prune_cross_edges)

relu = lambda x: np.maximum(x, 0)
ident = lambda x: x

def path(n, extra=()):
    A = np.zeros((n, n))
    for i in range(n - 1): A[i, i + 1] = A[i + 1, i] = 1
    for i, j in extra: A[i, j] = A[j, i] = 1
    return A

def weights(L, d_in=2, d_h=3):
    W, b = [], []
    for l in range(L):
        di = d_in if l == 0 else d_h
        W.append({"c": (np.arange(di * d_h).reshape(di, d_h) % 5 - 2) / 3.0,
                  "p": (np.arange(di * d_h).reshape(di, d_h)[::-1] % 7 - 3) / 4.0})
        b.append({"c": np.arange(d_h) / 10.0, "p": -np.arange(d_h) / 20.0})
    return W, b

X = (np.arange(12).reshape(6, 2) - 5) / 4.0
A_REL = {"c": path(6), "p": path(6, extra=[(0, 5)])}
ASSIGN = np.array([0, 0, 0, 1, 1, 1])

def test_gcn_norm_hand():
    A = path(2)                                  # one edge, both nodes get degree 2 -> every entry 1/2
    assert np.allclose(gcn_norm(A), np.full((2, 2), 0.5))

@pytest.mark.parametrize("L", [1, 2, 3])
def test_partitioned_equals_centralized_exactly(L):
    W, b = weights(L)
    acts = [relu] * (L - 1) + [ident]
    H_c = centralized_forward(A_REL, X, W, b, acts)
    H_p, log = partitioned_forward(A_REL, ASSIGN, X, W, b, acts)
    assert np.allclose(H_c, H_p, atol=1e-12)
    assert len(log["messages_per_layer"]) == L

def test_message_count_hand():
    # cut edge (2,3) in relation c and p; relation p also has edge (0,5):
    # cloudlet 0 needs {3, 5}, cloudlet 1 needs {2, 0}  -> 4 embeddings per layer
    W, b = weights(2)
    _, log = partitioned_forward(A_REL, ASSIGN, X, W, b, [relu, ident])
    assert log["messages_per_layer"] == [4, 4] and log["halo_sizes"] == {0: 2, 1: 2}

def test_single_cloudlet_has_no_messages_and_equals_centralized():
    W, b = weights(2)
    H_p, log = partitioned_forward(A_REL, np.zeros(6, int), X, W, b, [relu, ident])
    assert log["messages_per_layer"] == [0, 0]
    assert np.allclose(H_p, centralized_forward(A_REL, X, W, b, [relu, ident]))

def test_three_cloudlets_and_pruned_non_adjacent_edges():
    assign = np.array([0, 0, 1, 1, 2, 2])
    W, b = weights(2); acts = [relu, ident]
    # full graph: partitioned (all pairs may talk) == centralized
    H_p, _ = partitioned_forward(A_REL, assign, X, W, b, acts)
    assert np.allclose(H_p, centralized_forward(A_REL, X, W, b, acts))
    # edge (0,5) joins cloudlets 0 and 2.  If only (0,1) and (1,2) are adjacent, that edge is lost:
    pruned = prune_cross_edges(A_REL, assign, allowed_pairs={(0, 1), (1, 2)})
    assert pruned["p"][0, 5] == 0 and pruned["c"][1, 2] == 1
    H_pr, _ = partitioned_forward(pruned, assign, X, W, b, acts)
    assert np.allclose(H_pr, centralized_forward(pruned, X, W, b, acts))           # exact on the pruned graph
    assert not np.allclose(H_pr, centralized_forward(A_REL, X, W, b, acts))        # but differs from the full graph (SC3 degradation)
