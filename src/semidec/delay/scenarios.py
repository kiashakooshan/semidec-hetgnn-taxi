"""Delay accounting for the three inference scenarios of Sec. IV-D.  All times in milliseconds.

Only what the paper states is hard-wired (see constants.py).  Everything it leaves open is an EXPLICIT argument:
  * t_r_ms / t_cln_ms have no default (the paper gives no inter-cloudlet delay, and two different ad-hoc delays),
  * GNN-layer and message-passing-by-computation times are MEASURED callables of the graph size (benchmark.py),
  * uplink/downlink concurrency defaults to 1 packet time (docs/ASSUMPTIONS.md A-21).
The decentralized 'actual' delay of Fig. 7 cannot be defined from the text (the relaying protocol is not specified),
so the decentralized scenario is reported as the Theorem-1 bounds (A-22).
"""
from __future__ import annotations
from typing import Callable
from .constants import PAPER, PaperDelayConstants
from .theorem1 import theorem1_bounds, Bounds
from ..graph.neighborhood import Adj, l_hop_ball


def centralized_delay_ms(L: int, t_mp_ms: float, t_layer_ms: float, consts: PaperDelayConstants = PAPER,
                         n_serial_uplinks: int = 1, n_serial_downlinks: int = 1) -> float:
    """upload + L x (message passing by computation + GNN layer) + download   (bullet 1 of Sec. IV-D)."""
    if L < 1 or n_serial_uplinks < 1 or n_serial_downlinks < 1:
        raise ValueError("L and packet counts must be >= 1")
    pkt = consts.its_g5_packet_ms
    return n_serial_uplinks * pkt + L * (t_mp_ms + t_layer_ms) + n_serial_downlinks * pkt


def decentralized_delay_bounds_ms(adj: Adj, i: int, L: int, layer_time_fn: Callable[[int], float], t_r_ms: float,
                                  consts: PaperDelayConstants = PAPER, processing_term: str = "inside") -> Bounds:
    """Theorem-1 bounds for node i.  t_p = device slowdown (x100) times the measured layer time on i's L-hop subgraph."""
    n_local = 1 + len(l_hop_ball(adj, i, L))
    t_p = consts.device_slowdown_vs_cloud * layer_time_fn(n_local)
    return theorem1_bounds(adj, i, L, t_r_ms, t_p, processing_term)


def assignment_delay_ms(t_cln_ms: float, t_kmeans_ms: float) -> float:
    """Adaptive assignment (Sec. IV-D): send info to the neighbour + k-means there + send the result back."""
    return 2.0 * t_cln_ms + t_kmeans_ms


def semi_decentralized_delay_ms(L: int, log: dict, layer_time_fn: Callable[[int], float],
                                mp_time_fn: Callable[[int], float], t_cln_ms: float,
                                consts: PaperDelayConstants = PAPER, adaptive_ms: float = 0.0,
                                n_serial_uplinks: int = 1, n_serial_downlinks: int = 1) -> float:
    """Time of the SLOWEST cloudlet (Sec. IV-D bullet 3), layers being synchronous across cloudlets:

        upload + L x max_c [ t_mp(n_c) + 10 x t_layer(n_c) + t_cln x (received_c + sent_c) ] + download + adaptive

    n_c = own + halo nodes of cloudlet c (what it actually computes with); `log` comes from partitioned_forward.
    The inter-cloudlet link is treated as half-duplex: every node-message sent or received costs t_cln, in sequence.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    per_layer = []
    for c, own in log["own_sizes"].items():
        n_c = own + log["halo_sizes"][c]
        per_layer.append(mp_time_fn(n_c)
                         + consts.cloudlet_slowdown_vs_cloud * layer_time_fn(n_c)
                         + t_cln_ms * (log["halo_sizes"][c] + log["sent_sizes"][c]))
    pkt = consts.its_g5_packet_ms
    return n_serial_uplinks * pkt + L * max(per_layer) + n_serial_downlinks * pkt + adaptive_ms
