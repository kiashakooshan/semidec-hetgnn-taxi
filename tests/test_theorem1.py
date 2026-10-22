"""Hand-computed checks on the exact example graph of the paper's Fig. 15:
   i -- j, i -- k ; j -- j1,j2,j3 ; k -- k1..k4.   deg: i=2, j=4, k=5, leaves=1."""
import pytest
from semidec.graph.neighborhood import build_adj, hop_shells, degree
from semidec.delay.theorem1 import theorem1_bounds, two_hop_bounds_from_proof

# node ids: i=0 j=1 k=2 j1..j3=3,4,5 k1..k4=6,7,8,9
EDGES = [(0,1),(0,2),(1,3),(1,4),(1,5),(2,6),(2,7),(2,8),(2,9)]
adj = build_adj(10, EDGES)

def test_graph_structure():
    assert [degree(adj, x) for x in (0,1,2,3,6)] == [2,4,5,1,1]
    s = hop_shells(adj, 0, 3)
    assert s[0] == {1,2} and s[1] == {3,4,5,6,7,8,9} and s[2] == set()

def test_two_hop_from_proof_eq14():
    # relays j,k have degrees 4 and 5 ; t_r=1, t_p=0.1
    b = two_hop_bounds_from_proof(2, [4, 5], 1.0, 0.1)
    assert b.lower == pytest.approx(2*(4+5) + 0.3)      # 18.3
    assert b.upper == pytest.approx(2*(4+9) + 0.3)      # 26.3

def test_shell_reading_differs_from_proof_reading():
    # Eq.(14) with N_2(i) = exact 2-hop shell {j1..k4} (all degree 1):
    # this is what the Theorem statement literally says, and it does NOT equal the proof's numbers.
    L, t_r, t_p = 2, 1.0, 0.1
    b = theorem1_bounds(adj, 0, L, t_r, t_p, processing_term="inside")
    # l=1: w=2 : 2*(1*2 + max{4,5} + 0.2)=2*(7.2)=14.4 ; l=2: w=1 : (2*2 + 1 + 0.3)=5.3  => 2*... see below
    # printed form: 2*t_r*sum_l w*[l d_i + max + (l+1)t_p] = 2*(2*(2+5+0.2) + 1*(4+1+0.3)) = 2*(14.4+5.3)
    assert b.lower == pytest.approx(2 * (14.4 + 5.3))     # 39.4
    # upper: sums: l=1 sum=9 ; l=2 sum=7 -> 2*(2*(2+9+0.2) + (4+7+0.3))
    assert b.upper == pytest.approx(2 * (2*(2+9+0.2) + (4+7+0.3)))   # 2*(22.4+11.3)=67.4

def test_processing_term_outside():
    b = theorem1_bounds(adj, 0, 2, 1.0, 0.1, processing_term="outside")
    # comm part: 2*(2*(2+5) + 1*(4+1)) = 2*19 = 38 ; processing: 2*0.2 + 1*0.3 = 0.7
    assert b.lower == pytest.approx(38.7)

def test_lower_le_upper_and_monotone_in_L():
    prev = 0
    for L in range(1, 4):
        b = theorem1_bounds(adj, 0, L, 1.0, 0.1)
        assert b.lower <= b.upper
        assert b.lower > prev
        prev = b.lower

def test_printed_weight_sum_is_cubic_not_quadratic():
    # In Eq.(8) the term l*d_i is weighted by (L-l+1):  sum_{l=1}^{L} (L-l+1)*l = L(L+1)(L+2)/6  (cubic in L).
    for L in range(1, 12):
        assert sum((L-l+1)*l for l in range(1, L+1)) == L*(L+1)*(L+2)//6

def test_invalid_args():
    with pytest.raises(ValueError): theorem1_bounds(adj, 0, 0, 1, 1)
    with pytest.raises(ValueError): theorem1_bounds(adj, 0, 2, -1, 1)
    with pytest.raises(ValueError): theorem1_bounds(adj, 0, 2, 1, 1, processing_term="x")
