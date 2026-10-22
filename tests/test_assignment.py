import numpy as np
from semidec.graph.neighborhood import build_adj
from semidec.partition.assignment import (cross_edge_count, grid_assignment, boundary_nodes,
                                          shared_region, adaptive_reassign_pair,
                                          adaptive_assignment_all_pairs)

def test_cross_edges_and_boundary_hand_graph():
    # 0-1-2-3 path, CLN: [0,0,1,1] -> exactly one cross edge (1,2)
    adj = build_adj(4, [(0,1),(1,2),(2,3)])
    a = np.array([0,0,1,1])
    assert cross_edge_count(adj, a) == 1
    assert boundary_nodes(adj, a) == {1,2}
    assert shared_region(adj, a, 0, 1, hops=0) == {1,2}
    assert shared_region(adj, a, 0, 1, hops=1) == {0,1,2,3}

def test_grid_assignment():
    # bounding box 0..9 -> cell boundary at 4.5 in both axes; label = row*n_cols + col
    pos = np.array([[0,0],[9,0],[0,9],[9,9],[4.4,4.4],[4.6,4.6]], float)
    g = grid_assignment(pos, 2, 2)
    assert [int(x) for x in g] == [0,1,2,3,0,3]

def _two_blobs():
    # blob L: x in {0,1,2,3}, blob R: x in {5,6,7,8}; y in {0,1}.  proximity edges (dist<=1) inside blobs,
    # plus two bridges across the gap (3->5).  Intra-blob edges have length 1, bridges length 2.
    xs = [0,1,2,3,5,6,7,8]
    pos, idx = [], {}
    for x in xs:
        for y in (0,1):
            idx[(x,y)] = len(pos); pos.append((x,y))
    edges = []
    for (x,y),i in idx.items():
        for (x2,y2),j in idx.items():
            if i<j and abs(x-x2)+abs(y-y2)==1 and not (x==3 and x2==5) and not (x==5 and x2==3):
                edges.append((i,j))
    edges += [(idx[(3,0)],idx[(5,0)]), (idx[(3,1)],idx[(5,1)])]
    return np.array(pos,float), build_adj(len(pos), edges), idx

def test_adaptive_moves_misplaced_column_and_reduces_cut():
    pos, adj, idx = _two_blobs()
    # coverage line at x=5.5  -> column x=5 wrongly sits with blob L's CLN (0), rest of R is CLN 1
    assign = np.array([0 if p[0] < 5.5 else 1 for p in pos])
    before = cross_edge_count(adj, assign)
    res = adaptive_reassign_pair(adj, pos, assign, 0, 1, hops=1, seed=0)
    after = cross_edge_count(adj, res.assignment)
    assert before == 2          # (5,y)-(6,y) for y=0,1
    assert after == 2           # now the two bridges (3,y)-(5,y)
    # nodes outside the shared region never move
    outside = [i for i in range(len(pos)) if i not in res.region]
    assert np.array_equal(res.assignment[outside], assign[outside])
    # column x=5 moved to CLN 1 (blob R), blob L stays in CLN 0
    assert res.assignment[idx[(5,0)]] == 1 and res.assignment[idx[(3,0)]] == 0

def test_adaptive_is_deterministic_and_total():
    pos, adj, _ = _two_blobs()
    assign = np.array([0 if p[0] < 5.5 else 1 for p in pos])
    r1 = adaptive_assignment_all_pairs(adj, pos, assign, hops=1, seed=3)
    r2 = adaptive_assignment_all_pairs(adj, pos, assign, hops=1, seed=3)
    assert np.array_equal(r1, r2)
    assert set(np.unique(r1)) <= {0,1} and len(r1) == len(pos)
