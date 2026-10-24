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


# ---------- loader: must work whatever the downloaded file really is ----------
import os, zipfile
from semidec.data.zones import sniff, load_zones, find_id_col


def _gdf():
    return gpd.GeoDataFrame({"LocationID": [1, 2], "zone": ["a", "b"]},
                            geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs="EPSG:2263")


def _zip_shapefile(tmp_path, nested: bool):
    src = tmp_path / "src"; src.mkdir()
    _gdf().to_file(src / "taxi_zones.shp")
    z = tmp_path / "taxi_zones.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in os.listdir(src):
            zf.write(src / f, ("taxi_zones/" if nested else "") + f)
    return str(z)


@pytest.mark.parametrize("nested", [False, True])
def test_load_zipped_shapefile_root_or_nested(tmp_path, nested):
    g = load_zones(_zip_shapefile(tmp_path, nested), workdir=str(tmp_path / "out"))
    assert sorted(g["LocationID"]) == [1, 2] and g.crs is not None


def test_load_geoparquet_even_if_named_zip(tmp_path):
    p = tmp_path / "taxi_zones.zip"; _gdf().to_parquet(p)
    assert sniff(str(p))[0] == "parquet"
    assert sorted(load_zones(str(p))["LocationID"]) == [1, 2]


def test_html_page_saved_as_zip_gives_clear_error(tmp_path):
    p = tmp_path / "taxi_zones.zip"; p.write_text("<html>Access denied</html>")
    with pytest.raises(ValueError, match="HTML"):
        load_zones(str(p))


def test_find_id_col_variants():
    assert find_id_col(["OBJECTID", "location_id"]) == "location_id"
    assert find_id_col(["LocationID"]) == "LocationID"
    with pytest.raises(ValueError):
        find_id_col(["a", "b"])
