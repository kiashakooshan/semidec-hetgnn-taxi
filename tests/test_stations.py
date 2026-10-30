import numpy as np, pytest
from semidec.data.stations import (vicinity_indices, step_features, window, split_bounds, sample_starts,
                                   fit_stats, normalize, denormalize)

def test_vicinity_hand_and_ties():
    sim = np.array([[1, .2, .9, .5], [.2, 1, .1, .3], [.9, .1, 1, .4], [.5, .3, .4, 1]])
    assert vicinity_indices(sim, 3).tolist() == [[0, 2, 3], [1, 3, 0], [2, 0, 3], [3, 0, 2]]
    assert vicinity_indices(np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]), 2)[0].tolist() == [0, 1]   # tie -> lower index
    with pytest.raises(ValueError): vicinity_indices(sim, 5)
    with pytest.raises(ValueError): vicinity_indices(np.ones((2, 3)), 1)

PICK = np.array([[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]], float)       # T=4, N=3
DROP = PICK + 100
VIC = np.array([[0, 2], [1, 0], [2, 1]])                                                  # k=2

def test_step_features_layout_hand():
    F = step_features(PICK, DROP, VIC)
    assert F.shape == (4, 3, 4)
    assert F[0, 0].tolist() == [10, 30, 110, 130]            # node 0 at t=0: pick of (0,2) then drop of (0,2)
    assert F[1, 1].tolist() == [21, 11, 121, 111]

def test_window_is_time_major_and_aligned():
    F = step_features(PICK, DROP, VIC)
    x, y = window(F, t=2, P=2, Q=2)
    assert x.shape == (3, 8) and y.shape == (3, 8)
    assert x[0].tolist() == F[0, 0].tolist() + F[1, 0].tolist()      # steps 0,1 for node 0, one step after the other
    assert y[0].tolist() == F[2, 0].tolist() + F[3, 0].tolist()      # target = steps 2,3
    with pytest.raises(IndexError): window(F, 1, 2, 2)
    with pytest.raises(IndexError): window(F, 3, 2, 2)

def test_split_bounds_and_sample_counts_hand():
    b = split_bounds(20, val_steps=4, test_steps=4)
    assert b == dict(train=(0, 12), val=(12, 16), test=(16, 20))
    P, Q = 3, 2
    assert sample_starts(b["train"], P, Q).tolist() == list(range(3, 11))       # 8 samples, last target [10,12)
    assert sample_starts(b["val"], P, Q).tolist() == [12, 13, 14]
    assert sample_starts(b["test"], P, Q).tolist() == [16, 17, 18]
    # no leakage: every target step of a split lies inside that split, and splits' targets are disjoint
    seen = set()
    for name, rg in b.items():
        steps = {t + j for t in sample_starts(rg, P, Q) for j in range(Q)}
        assert all(rg[0] <= s < rg[1] for s in steps)
        assert not (steps & seen); seen |= steps
    with pytest.raises(ValueError): split_bounds(10, 5, 5)

def test_paper_sizes():
    b = split_bounds(4368, 672, 672)
    assert b == dict(train=(0, 3024), val=(3024, 3696), test=(3696, 4368))
    assert len(sample_starts(b["train"], 12, 12)) == 3024 - 12 - 12 + 1 == 3001

def test_normalization_uses_train_only_and_round_trips():
    pick = np.arange(40, dtype=float).reshape(10, 4); drop = pick * 2
    st = fit_stats(pick, drop, (0, 6))
    assert st.mean_pick == pytest.approx(pick[:6].mean()) and st.mean_pick != pytest.approx(pick.mean())   # val/test excluded
    F = step_features(pick, drop, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    Z = normalize(F, st, k=2)
    assert np.allclose(denormalize(Z, st, k=2), F)
    x, y = window(Z, 6, 2, 2)                                                       # flattened windows invert too
    assert np.allclose(denormalize(y, st, k=2), window(F, 6, 2, 2)[1])
    with pytest.raises(ValueError): fit_stats(np.ones((5, 2)), drop[:5, :2], (0, 5))
