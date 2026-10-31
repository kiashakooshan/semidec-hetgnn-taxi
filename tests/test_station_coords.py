import numpy as np, pandas as pd, pytest
from semidec.data.station_coords import (load_taxi_coords, pairwise_km, gaussian_kernel, compare_kernel,
                                         own_zone_rank, TAXI_COLS)

def test_pairwise_km_one_degree_of_latitude():
    D = pairwise_km([40.0, 41.0], [-74.0, -74.0])
    assert D[0, 1] == pytest.approx(111.195, abs=0.01) and D[0, 0] == 0

def test_kernel_hand_values():
    d1 = 1.0 / 111.195                                              # degrees of latitude per km
    D = pairwise_km([40.0, 40.0 + d1, 40.0 + 3 * d1], [-74.0] * 3)  # 0, 1 km, 3 km on a line
    W = gaussian_kernel(D)
    assert W[0, 1] == pytest.approx(np.exp(-1.0), abs=1e-4)          # 0.3679
    assert W[0, 2] == 0.0 and W[1, 2] == pytest.approx(np.exp(-4.0), abs=1e-3) or W[1, 2] == 0.0   # 2 km -> at the cutoff
    assert np.allclose(np.diag(W), 1.0) and np.allclose(W, W.T)

def test_compare_kernel_identical_and_perturbed():
    D = np.array([[0, .5, 1.5], [.5, 0, 1.0], [1.5, 1.0, 0]])
    W = gaussian_kernel(D)
    r = compare_kernel(W, W)
    assert r["only_pred"] == r["only_true"] == 0 and r["dist_err_m_max"] == 0 and r["nnz_common"] == 6
    D2 = D.copy(); D2[0, 1] = D2[1, 0] = 0.51                       # move one pair by 10 m
    r2 = compare_kernel(gaussian_kernel(D2), W)
    assert r2["dist_err_m_max"] == pytest.approx(10.0, abs=0.01)
    with pytest.raises(ValueError): compare_kernel(W, W[:2, :2])

def test_own_zone_rank_hand():
    t = np.arange(10, dtype=float)
    zones = np.stack([t, np.sin(t), -t], 1)                           # zone 0 = ramp, 1 = sine, 2 = decreasing ramp
    stations = np.stack([2 * t + 1, -t], 1)                           # station 0 ~ zone 0 ; station 1 ~ zone 2
    assert own_zone_rank(stations, zones, np.array([0, 2])).tolist() == [1, 1]
    assert own_zone_rank(stations, zones, np.array([2, 0])).tolist() == [3, 3]       # wrong zones rank last
    assert own_zone_rank(stations, zones, np.array([-1, 0])).tolist()[0] == 0        # outside every zone

def test_loader_validation(tmp_path):
    good = pd.DataFrame({"station_id": [1, 2], "lat": [40.7, 40.71], "lon": [-74.0, -73.99], "x_km": [0., 1.], "y_km": [0., 1.], "n_bike_anchors": [3, 4]})
    p = tmp_path / "t.csv"; good.to_csv(p, index=False)
    assert len(load_taxi_coords(str(p), n_expected=2)) == 2
    with pytest.raises(ValueError): load_taxi_coords(str(p), n_expected=266)
    bad = good.copy(); bad.loc[1, "lat"] = 10.0; bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="NYC"): load_taxi_coords(str(p), n_expected=2)
    bad = good.copy(); bad["station_id"] = [2, 1]; bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="1..N"): load_taxi_coords(str(p), n_expected=2)

def test_the_delivered_file_loads():
    df = load_taxi_coords("data/external/station_coordinates_taxi266.csv")
    assert list(df.columns) == TAXI_COLS and len(df) == 266


def test_rectangular_kernel_has_no_forced_diagonal():
    D = np.array([[0.0, 1.0, 3.0], [3.0, 0.5, 1.5]])                  # 2 taxi x 3 bike distances (not square)
    W = gaussian_kernel(D, set_diag=False)
    assert W[0, 0] == 1.0 and W[1, 0] == 0.0 and W[0, 2] == 0.0       # d=0 -> 1 by the formula; >2 km -> 0
    assert W[1, 1] == pytest.approx(np.exp(-0.25)) and W[1, 2] == pytest.approx(np.exp(-2.25))
    r = compare_kernel(W, W)
    assert r["nnz_common"] == 4 and r["only_pred"] == 0 and r["only_true"] == 0
