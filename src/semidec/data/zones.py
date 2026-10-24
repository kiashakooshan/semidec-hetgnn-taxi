"""Zone centroids from the official TLC taxi-zone shapefile.

The shapefile has several polygons for some LocationIDs, so polygons are first dissolved
(unioned) by id and only then is the centroid taken.  x, y are in the shapefile's projected CRS
(EPSG:2263, feet) and lon/lat (EPSG:4326) are added when the CRS is known.
"""
from __future__ import annotations
import pandas as pd


def zone_centroids(gdf, id_col: str = "LocationID") -> pd.DataFrame:
    if id_col not in gdf.columns:
        raise ValueError(f"missing column {id_col}")
    d = gdf[[id_col, "geometry"]].dissolve(by=id_col)          # union polygons that share an id
    cen = d.geometry.centroid
    out = pd.DataFrame({"zone_id": d.index.astype(int), "x": cen.x.values, "y": cen.y.values})
    if gdf.crs is not None:
        ll = cen.set_crs(gdf.crs).to_crs(4326)
        out["lon"], out["lat"] = ll.x.values, ll.y.values
    return out.sort_values("zone_id").reset_index(drop=True)


# ---------------------------------------------------------------- loading the official zone file
import os
import re
import zipfile


def sniff(path: str) -> tuple[str, bytes]:
    """Real file type from its first bytes (file extensions on downloads can lie)."""
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:2] == b"PK":
        return "zip", head
    if head[:4] == b"PAR1":
        return "parquet", head
    return "unknown", head


def find_id_col(columns) -> str:
    for c in columns:
        if re.sub(r"[_\s]", "", str(c)).lower() == "locationid":
            return c
    raise ValueError(f"no LocationID-like column in {list(columns)}")


def load_zones(path: str, workdir: str | None = None):
    """Read the TLC zone geometry whether it is a zipped shapefile or a (Geo)Parquet file."""
    import geopandas as gpd
    kind, head = sniff(path)
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            shp = [n for n in names if n.lower().endswith(".shp")]
            if not shp:
                raise ValueError(f"zip contains no .shp file. Contents: {names[:20]}")
            out = workdir or (os.path.splitext(path)[0] + "_extracted")
            z.extractall(out)
        return gpd.read_file(os.path.join(out, shp[0]))
    if kind == "parquet":
        try:
            return gpd.read_parquet(path)
        except Exception as e:                                   # plain parquet without geometry metadata
            import pandas as pd
            cols = list(pd.read_parquet(path).columns)
            raise ValueError(f"Parquet is not GeoParquet ({e}). Columns: {cols}")
    raise ValueError(f"Unrecognized file (first bytes {head!r}). If it starts with '<' it is probably an HTML "
                     "page saved instead of the file: download it again.")
