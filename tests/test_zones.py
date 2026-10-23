import pytest
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box
from semidec.data.zones import zone_centroids

def test_dissolve_duplicate_ids_then_centroid():
    gdf = gpd.GeoDataFrame({"LocationID": [1, 1, 2]},
                           geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 12, 2)])
    df = zone_centroids(gdf)
    assert list(df.zone_id) == [1, 2]
    assert df.x.tolist() == pytest.approx([1.0, 11.0]) and df.y.tolist() == pytest.approx([0.5, 1.0])

def test_lonlat_for_ny_state_plane():
    # EPSG:2263 (feet): a block near lower Manhattan/Brooklyn must land inside the NYC box
    gdf = gpd.GeoDataFrame({"LocationID": [7]}, geometry=[box(990000, 200000, 991000, 201000)], crs="EPSG:2263")
    r = zone_centroids(gdf).iloc[0]
    assert 40.4 < r.lat < 41.0 and -74.3 < r.lon < -73.6
