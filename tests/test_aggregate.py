import numpy as np, pytest
from semidec.data.aggregate import SlotGrid, count_events

D = lambda s: np.datetime64(s)

def test_paper_window_has_4368_half_hour_slots():
    g = SlotGrid(D("2016-04-01T00:00:00"), D("2016-07-01T00:00:00"), 30)
    assert g.n_slots == 91 * 48 == 4368            # 30+31+30 days
    assert SlotGrid(g.start, g.end, 60).n_slots == 2184

def test_slot_boundaries_hand_checked():
    g = SlotGrid(D("2016-04-01T00:00:00"), D("2016-04-02T00:00:00"), 30)
    ts = np.array(["2016-04-01T00:00:00", "2016-04-01T00:29:59", "2016-04-01T00:30:00",
                   "2016-04-01T23:59:59", "2016-04-02T00:00:00", "2016-03-31T23:59:59", "NaT"], "datetime64[s]")
    assert list(g.slot_index(ts)) == [0, 0, 1, 47, -1, -1, -1]

def test_count_events_and_reporting():
    g = SlotGrid(D("2016-04-01T00:00:00"), D("2016-04-02T00:00:00"), 30)
    ts = np.array(["2016-04-01T00:00:00", "2016-04-01T00:29:59", "2016-04-01T00:30:00", "2016-04-01T23:59:59",
                   "2016-04-02T00:00:00",            # outside (end exclusive)
                   "2016-04-01T00:10:00",            # zone 0 -> bad
                   "2016-04-01T00:10:00"],           # zone 4 > n_zones=3 -> bad
                  "datetime64[s]")
    zone = np.array([1, 1, 1, 3, 2, 0, 4])
    c, s = count_events(ts, zone, g, n_zones=3)
    assert c.shape == (48, 3)
    assert c[0, 0] == 2 and c[1, 0] == 1 and c[47, 2] == 1 and c.sum() == 4
    assert s == dict(n_total=7, n_counted=4, n_outside_window=1, n_bad_zone_inside_window=2)

def test_invalid_window():
    with pytest.raises(ValueError):
        SlotGrid(D("2016-04-01T00:00:00"), D("2016-04-01T00:45:00"), 30)
