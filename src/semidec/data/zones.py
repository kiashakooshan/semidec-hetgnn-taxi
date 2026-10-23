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
