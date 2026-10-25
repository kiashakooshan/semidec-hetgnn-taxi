"""Hand-computed delay accounting (ms).  Timing callables are simple deterministic functions of the node count."""
import numpy as np, pytest
from semidec.delay.scenarios import (centralized_delay_ms, decentralized_delay_bounds_ms,
                                     semi_decentralized_delay_ms, assignment_delay_ms)
from semidec.delay.constants import PAPER
from semidec.graph.neighborhood import build_adj
from semidec.inference.partitioned import partitioned_forward
from tests.test_partitioned import A_REL, ASSIGN, X, weights, relu, ident

def test_paper_constants_are_the_stated_ones():
    assert PAPER.its_g5_packet_ms == 3.3 and PAPER.adhoc_source_ms == pytest.approx(23.55)
    assert PAPER.adhoc_relay_ms == pytest.approx(18.65)
    assert (PAPER.device_slowdown_vs_cloud, PAPER.cloudlet_slowdown_vs_cloud) == (100.0, 10.0)

def test_centralized_hand():
    assert centralized_delay_ms(3, 0.5, 2.0) == pytest.approx(3.3 + 3 * 2.5 + 3.3)               # 14.1
    assert centralized_delay_ms(3, 0.5, 2.0, n_serial_uplinks=4) == pytest.approx(13.2 + 7.5 + 3.3)   # 24.0

def test_assignment_delay():
    assert assignment_delay_ms(10.0, 4.0) == 24.0

def test_semi_decentralized_hand_on_two_cloudlets():
    W, b = weights(2)
    _, log = partitioned_forward(A_REL, ASSIGN, X, W, b, [relu, ident])
    # each cloudlet: own 3 + halo 2 = 5 nodes ; recv 2 ; sent 2.  mp = 0.1*5, layer = 10 x (1.0*5), cln = 10 x (2+2)
    per_layer = 0.5 + 50.0 + 40.0
    t = semi_decentralized_delay_ms(2, log, lambda n: 1.0 * n, lambda n: 0.1 * n, t_cln_ms=10.0)
    assert t == pytest.approx(3.3 + 2 * per_layer + 3.3)                                          # 187.6
    t_ad = semi_decentralized_delay_ms(2, log, lambda n: 1.0 * n, lambda n: 0.1 * n, 10.0,
                                       adaptive_ms=assignment_delay_ms(10.0, 4.0))
    assert t_ad == pytest.approx(187.6 + 24.0)

def test_slowest_cloudlet_decides():
    # layer time 1.0 ms, mp 0, t_cln 5, cloudlet slowdown 10 -> compute 10 ms per cloudlet
    log = dict(own_sizes={0: 4, 1: 2}, halo_sizes={0: 1, 1: 3}, sent_sizes={0: 3, 1: 1})
    # c0: 10 + 5*(1+3) = 30 ; c1: 10 + 5*(3+1) = 30
    t = semi_decentralized_delay_ms(1, log, lambda n: 1.0, lambda n: 0.0, t_cln_ms=5.0)
    assert t == pytest.approx(3.3 + 30.0 + 3.3)
    log["sent_sizes"][1] = 5                     # c1 now: 10 + 5*(3+5) = 50 -> it becomes the slowest
    assert semi_decentralized_delay_ms(1, log, lambda n: 1.0, lambda n: 0.0, 5.0) == pytest.approx(3.3 + 50.0 + 3.3)

def test_decentralized_bounds_reuse_theorem1_on_fig15_graph():
    adj = build_adj(10, [(0,1),(0,2),(1,3),(1,4),(1,5),(2,6),(2,7),(2,8),(2,9)])
    # t_p = 100 * 0.001 = 0.1 ; t_r = 1 -> same numbers as tests/test_theorem1.py for L=2
    b = decentralized_delay_bounds_ms(adj, 0, 2, lambda n: 0.001, t_r_ms=1.0)
    assert b.lower == pytest.approx(39.4) and b.upper == pytest.approx(67.4)

def test_invalid():
    with pytest.raises(ValueError): centralized_delay_ms(0, 1, 1)
