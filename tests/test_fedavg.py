import pytest
torch = pytest.importorskip("torch")
from semidec.federated.fedavg import average_state_dicts, server_free_round

S = lambda v: {"w": torch.tensor([v], dtype=torch.float32), "n": torch.tensor([7])}

def test_average_hand():
    out = average_state_dicts([S(0.0), S(3.0)])
    assert out["w"].item() == pytest.approx(1.5) and out["n"].item() == 7          # int buffer copied
    out = average_state_dicts([S(0.0), S(3.0)], weights=[1, 3])
    assert out["w"].item() == pytest.approx(2.25)                                    # (0*1 + 3*3)/4

def test_server_free_round_on_a_path_of_three_clients():
    states = {0: S(0.0), 1: S(3.0), 2: S(9.0)}
    nb = {0: {1}, 1: {0, 2}, 2: {1}}
    new = server_free_round(states, nb)
    assert [new[c]["w"].item() for c in (0, 1, 2)] == pytest.approx([1.5, 4.0, 6.0])
    new = server_free_round(states, nb, sizes={0: 1, 1: 1, 2: 2})
    assert new[1]["w"].item() == pytest.approx((0 + 3 + 18) / 4)
    assert new[2]["w"].item() == pytest.approx((3 + 18) / 3)

def test_asymmetric_adjacency_rejected():
    with pytest.raises(ValueError):
        server_free_round({0: S(0.0), 1: S(1.0)}, {0: {1}, 1: set()})

def test_bad_weights():
    with pytest.raises(ValueError):
        average_state_dicts([S(0.0)], weights=[0])
