import numpy as np, pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from semidec.model.hetgnn_lstm import HetGCNLayer, HetGNN, HetGNNLSTM, make_optimizer, RELATIONS
from semidec.inference.partitioned import gcn_norm, centralized_forward, partitioned_forward


def dense_to_edges(A):
    r, c = np.nonzero(A)
    return torch.tensor(np.stack([r, c]), dtype=torch.long), torch.tensor(A[r, c], dtype=torch.float64)


def graph6():
    def sym(edges, w=None):
        A = np.zeros((6, 6))
        for k, (i, j) in enumerate(edges):
            A[i, j] = A[j, i] = 1.0 if w is None else w[k]
        return A
    return {"connectivity": sym([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            "proximity": sym([(0, 2), (2, 3), (3, 5)], w=[0.5, 2.0, 0.25]),
            "destination": sym([(0, 5), (1, 4)])}


X6 = torch.tensor((np.arange(24).reshape(6, 4) - 11) / 7.0, dtype=torch.float64)


def test_shapes_and_validation():
    torch.manual_seed(0)
    m = HetGNNLSTM(lstm_layers=2)
    A = graph6()
    ei = {r: dense_to_edges(A[r])[0] for r in RELATIONS}
    assert m(torch.zeros(6, 216), ei).shape == (6, 216)
    with pytest.raises(ValueError):
        HetGNNLSTM(in_dim=100, steps=12)


def test_het_layer_equals_dense_formula():
    torch.manual_seed(1)
    A = graph6()
    layer = HetGCNLayer(4, 3).double()
    ei, ew = {}, {}
    for r in RELATIONS:
        ei[r], ew[r] = dense_to_edges(A[r])
    out = layer(X6, ei, ew).detach().numpy()
    ref = sum(gcn_norm(A[r]) @ (X6.numpy() @ layer.convs[r].lin.weight.detach().numpy().T)
              + layer.convs[r].bias.detach().numpy() for r in RELATIONS)
    assert np.allclose(out, ref, atol=1e-10)


def test_real_model_equals_numpy_centralized_and_partitioned():
    """Algorithm 1 on the REAL torch hetGNN: centralized == numpy centralized == 2-cloudlet partitioned."""
    torch.manual_seed(2)
    A = graph6()
    gnn = HetGNN(4, 5, 3, n_layers=3).double()
    ei, ew = {}, {}
    for r in RELATIONS:
        ei[r], ew[r] = dense_to_edges(A[r])
    out_t = gnn(X6, ei, ew).detach().numpy()
    W = [{r: layer.convs[r].lin.weight.detach().numpy().T for r in RELATIONS} for layer in gnn.layers]
    b = [{r: layer.convs[r].bias.detach().numpy() for r in RELATIONS} for layer in gnn.layers]
    acts = [lambda x: np.maximum(x, 0), lambda x: np.maximum(x, 0), lambda x: x]
    assert np.allclose(out_t, centralized_forward(A, X6.numpy(), W, b, acts), atol=1e-10)
    for assign in ([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2]):
        out_p, log = partitioned_forward(A, np.array(assign), X6.numpy(), W, b, acts)
        assert np.allclose(out_t, out_p, atol=1e-10) and len(log["messages_per_layer"]) == 3


def test_gradients_reach_every_parameter():
    torch.manual_seed(3)
    m = HetGNNLSTM(lstm_layers=2)
    A = graph6()
    ei = {r: dense_to_edges(A[r])[0] for r in RELATIONS}
    loss = torch.nn.functional.mse_loss(m(torch.ones(6, 216), ei), torch.zeros(6, 216))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_optimizer_matches_table_ii():
    opt = make_optimizer(HetGNNLSTM(lstm_layers=2))
    g, l = opt.param_groups
    assert (g["lr"], g["weight_decay"], l["lr"]) == (0.005, 0.001, 0.0001)
    assert isinstance(opt, torch.optim.Adam)


def test_paper_sized_model_builds_with_10_lstm_layers():
    m = HetGNNLSTM()
    assert m.lstm.num_layers == 10 and len(m.gnn.layers) == 3
