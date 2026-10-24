import numpy as np, pytest
from semidec.partition.clustering import kmedoids, fcm
from semidec.partition.assignment import adaptive_reassign_pair, cross_edge_count
from tests.test_assignment import _two_blobs

PTS = np.array([[0.], [1.], [2.], [10.], [11.], [12.]])

def test_kmedoids_hand():
    labels, med = kmedoids(PTS, 2)
    assert sorted(med.tolist()) == [1, 4]                 # medoids are the middle points 1.0 and 11.0
    assert labels.tolist()[:3] == [labels[0]] * 3 and labels.tolist()[3:] == [labels[3]] * 3 and labels[0] != labels[3]

def test_fcm_hand():
    labels, centers, U = fcm(PTS, 2)
    assert np.allclose(U.sum(1), 1.0)
    assert labels[0] == labels[1] == labels[2] != labels[3] == labels[4] == labels[5]
    assert sorted(centers[:, 0])[0] == pytest.approx(1.0, abs=0.5) and sorted(centers[:, 0])[1] == pytest.approx(11.0, abs=0.5)

def test_deterministic():
    assert np.array_equal(kmedoids(PTS)[0], kmedoids(PTS)[0]) and np.array_equal(fcm(PTS)[0], fcm(PTS)[0])

@pytest.mark.parametrize("method", ["kmeans", "kmedoids", "fcm"])
def test_all_methods_keep_invariants(method):
    pos, adj, idx = _two_blobs()
    assign = np.array([0 if p[0] < 5.5 else 1 for p in pos])
    res = adaptive_reassign_pair(adj, pos, assign, 0, 1, hops=1, method=method)
    assert set(np.unique(res.assignment)) <= {0, 1} and len(res.assignment) == len(pos)
    outside = [i for i in range(len(pos)) if i not in res.region]
    assert np.array_equal(res.assignment[outside], assign[outside])        # nodes outside V_b never move

@pytest.mark.parametrize("method", ["kmeans", "fcm"])
def test_kmeans_and_fcm_do_not_increase_the_cut_on_two_blobs(method):
    pos, adj, idx = _two_blobs()
    assign = np.array([0 if p[0] < 5.5 else 1 for p in pos])
    res = adaptive_reassign_pair(adj, pos, assign, 0, 1, hops=1, method=method)
    assert cross_edge_count(adj, res.assignment) <= cross_edge_count(adj, assign) == 2

def test_kmedoids_parkjun_init_documented_failure_mode():
    """FINDING (docs/PAPER_ISSUES.md P15): Park-Jun initialisation picks the two most central points
    (5,0),(5,1); they differ only in y, so the split is along y and the cut grows from 2 to 6.
    The paper's claim that the clustering criterion matters little (Fig. 11) is therefore not universal."""
    pos, adj, idx = _two_blobs()
    assign = np.array([0 if p[0] < 5.5 else 1 for p in pos])
    res = adaptive_reassign_pair(adj, pos, assign, 0, 1, hops=1, method="kmedoids")
    assert (res.cross_before, res.cross_after) == (2, 6)

def test_unknown_method():
    pos, adj, _ = _two_blobs()
    with pytest.raises(ValueError):
        adaptive_reassign_pair(adj, pos, np.array([0 if p[0] < 5.5 else 1 for p in pos]), 0, 1, method="x")
