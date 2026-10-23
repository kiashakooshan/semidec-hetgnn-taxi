"""Zone centroids from the official taxi_zones.zip (put it in data/raw/).
    python scripts/build_zone_centroids.py
"""
import argparse, os, sys
import geopandas as gpd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from semidec.data.zones import zone_centroids

ap = argparse.ArgumentParser()
ap.add_argument("--zip", default="data/raw/taxi_zones.zip")
ap.add_argument("--out", default="data/processed/zone_centroids.csv")
a = ap.parse_args()
if not os.path.exists(a.zip):
    sys.exit(f"File not found: {a.zip}\nDownload the official zone shapefile and put it there:\n"
             "  https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip\n(keep it as a .zip, do not extract)")
gdf = gpd.read_file(f"zip://{a.zip}")
print("columns:", list(gdf.columns), "| rows:", len(gdf), "| crs:", gdf.crs)
df = zone_centroids(gdf, "LocationID")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print("zones:", len(df), "| id range:", df.zone_id.min(), "-", df.zone_id.max()); print(df.head())
