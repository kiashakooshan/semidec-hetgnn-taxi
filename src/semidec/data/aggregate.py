"""Event -> (time slot, zone) counting for the TLC trip records.

Pure functions (no file I/O) so that they can be unit-tested by hand.
Window is half-open: start <= t < end.  Nothing is dropped silently: every event that is not
counted is reported in `stats`.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SlotGrid:
    start: np.datetime64          # inclusive
    end: np.datetime64            # exclusive
    slot_minutes: int

    def __post_init__(self):
        span = (np.datetime64(self.end, "s") - np.datetime64(self.start, "s")).astype(np.int64)
        if self.slot_minutes <= 0 or span <= 0 or span % (self.slot_minutes * 60) != 0:
            raise ValueError("window must be a positive multiple of slot_minutes")

    @property
    def n_slots(self) -> int:
        span = (np.datetime64(self.end, "s") - np.datetime64(self.start, "s")).astype(np.int64)
        return int(span // (self.slot_minutes * 60))

    def slot_index(self, ts: np.ndarray) -> np.ndarray:
        """Slot number of each timestamp; -1 if NaT or outside [start, end)."""
        ts = np.asarray(ts).astype("datetime64[s]")
        sec = (ts - np.datetime64(self.start, "s")).astype(np.int64)
        idx = sec // (self.slot_minutes * 60)
        bad = np.isnat(ts) | (sec < 0) | (idx >= self.n_slots)
        return np.where(bad, -1, idx)

    def slot_starts(self) -> np.ndarray:
        return np.datetime64(self.start, "s") + np.arange(self.n_slots) * np.timedelta64(self.slot_minutes * 60, "s")


def count_events(ts: np.ndarray, zone: np.ndarray, grid: SlotGrid, n_zones: int):
    """Return (counts[T, n_zones], stats). Zone ids are 1..n_zones, column = id-1."""
    ts = np.asarray(ts)
    zone = np.asarray(zone)
    if ts.shape != zone.shape:
        raise ValueError("ts and zone must have the same shape")
    slot = grid.slot_index(ts)
    out_window = slot < 0
    bad_zone = (zone < 1) | (zone > n_zones)
    ok = ~out_window & ~bad_zone
    T = grid.n_slots
    flat = slot[ok].astype(np.int64) * n_zones + (zone[ok].astype(np.int64) - 1)
    counts = np.bincount(flat, minlength=T * n_zones).reshape(T, n_zones).astype(np.int64)
    stats = dict(n_total=int(ts.size), n_counted=int(ok.sum()),
                 n_outside_window=int(out_window.sum()),
                 n_bad_zone_inside_window=int((bad_zone & ~out_window).sum()))
    assert stats["n_counted"] == counts.sum()
    return counts, stats
