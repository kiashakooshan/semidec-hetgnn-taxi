"""Delay constants that the paper states explicitly (Section IV-D).  Units: milliseconds.

Nothing here is invented: every value carries the sentence it comes from.
Quantities the paper does NOT give (GNN layer processing time, tr for the
bounds, ...) are NOT defaulted here; they must be measured or passed in.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperDelayConstants:
    # "we adopt a packet transmission delay of 3.3 ms as reported in [53]"  (taxi <-> server/cloudlet, ITS-G5)
    its_g5_packet_ms: float = 3.3
    # "(16.55+7) ms for source nodes" -- ad hoc network, from [54]
    adhoc_source_ms: float = 16.55 + 7.0
    # "(11.65+7) ms for relay nodes"  -- ad hoc network, from [54]
    adhoc_relay_ms: float = 11.65 + 7.0
    # "a decentralized device is assumed to have a 100 times longer processing time
    #  as compared to a centralized cloud server"
    device_slowdown_vs_cloud: float = 100.0
    # "a cloudlet ... is assumed to have an order of magnitude slower computation
    #  as compared to the cloud server"
    cloudlet_slowdown_vs_cloud: float = 10.0


PAPER = PaperDelayConstants()
