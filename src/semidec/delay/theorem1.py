"""Theorem 1 (decentralized GNN inference delay bounds), Eq. (8) and Appendix Eq. (14)-(16).

The paper is internally inconsistent about where the processing term (l+1)*t_p sits
and about what N_l(i) means (see docs/PAPER_ISSUES.md, P1-P3).  We therefore expose
every reading explicitly instead of silently picking one:

  processing_term="inside"  : Eq. (8) and Eq. (16) as printed:
        2 t_r * sum_l (L-l+1) [ l d_i + max/sum_x d_x + (l+1) t_p ]
  processing_term="outside" : Eq. (14)/(15) form (what the 2-hop derivation gives):
        sum_l (L-l+1) { 2 t_r [ l d_i + max/sum_x d_x ] + (l+1) t_p }

N_l(i) here is the shell of nodes at EXACTLY l hops from i (the definition used by the
statement of Theorem 1).  The 2-hop proof instead uses the degrees of the 1-hop relays
j,k; that reading is provided by `two_hop_bounds_from_proof`.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..graph.neighborhood import Adj, degree, hop_shells


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float


def _check(L: int, t_r: float, t_p: float):
    if L < 1:
        raise ValueError("L must be >= 1")
    if t_r < 0 or t_p < 0:
        raise ValueError("delays must be non-negative")


def theorem1_bounds(adj: Adj, i: int, L: int, t_r: float, t_p: float,
                    processing_term: str = "inside") -> Bounds:
    _check(L, t_r, t_p)
    if processing_term not in ("inside", "outside"):
        raise ValueError("processing_term must be 'inside' or 'outside'")
    shells = hop_shells(adj, i, L)
    d_i = degree(adj, i)
    lo = up = 0.0
    for l in range(1, L + 1):
        w = L - l + 1                      # hop-l messages are exchanged (L-l+1) times
        degs = [degree(adj, x) for x in shells[l - 1]]
        mx = max(degs) if degs else 0      # empty shell -> 0
        sm = sum(degs)
        if processing_term == "inside":
            lo += w * 2 * t_r * (l * d_i + mx + (l + 1) * t_p)
            up += w * 2 * t_r * (l * d_i + sm + (l + 1) * t_p)
        else:
            lo += w * (2 * t_r * (l * d_i + mx) + (l + 1) * t_p)
            up += w * (2 * t_r * (l * d_i + sm) + (l + 1) * t_p)
    return Bounds(lo, up)


def two_hop_bounds_from_proof(d_i: int, relay_degrees: list[int], t_r: float, t_p: float) -> Bounds:
    """Eq. (14) as the Appendix derives it, using the degrees of the 1-hop relays j,k.

    lower: relays work fully concurrently  -> max over relays
    upper: relays work sequentially        -> sum over relays
    """
    _check(2, t_r, t_p)
    if not relay_degrees:
        raise ValueError("need at least one relay")
    lo = 2 * t_r * (2 * d_i + max(relay_degrees)) + 3 * t_p
    up = 2 * t_r * (2 * d_i + sum(relay_degrees)) + 3 * t_p
    return Bounds(lo, up)
