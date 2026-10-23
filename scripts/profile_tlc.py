"""Profile one TLC yellow-taxi PARQUET file WITHOUT loading it fully into memory.

Usage:
    python scripts/profile_tlc.py data/raw/yellow_tripdata_2016-04.parquet
    python scripts/profile_tlc.py data/raw/yellow_tripdata_2016-04.parquet --sample 2000

Prints a report and writes:
    results/profile_<file>.json     (small; send/paste this)
    results/sample_<file>.csv       (first N rows; small enough to upload)

It answers, from the REAL file (nothing is assumed):
  1. which columns exist and their types
  2. are there pickup/dropoff longitude+latitude columns, and are the values plausible for NYC?
  3. are there taxi/driver/vehicle identifier columns? (needed for 'node = taxi')
  4. date range, rows per day, null counts
"""
import argparse, csv, json, os, re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NYC_LAT = (40.4, 41.0)
NYC_LON = (-74.3, -73.6)
ID_HINT = re.compile(r"(medallion|hack|licen[cs]e|vehicle|driver|taxi_?id|cab)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=500_000)
    a = ap.parse_args()

    pf = pq.ParquetFile(a.path)
    schema = pf.schema_arrow
    cols = [(f.name, str(f.type)) for f in schema]
    n_total = pf.metadata.num_rows
    names = [c for c, _ in cols]

    lat_cols = [c for c in names if re.search(r"lat", c, re.I)]
    lon_cols = [c for c in names if re.search(r"lon", c, re.I)]
    dt_cols = [c for c in names if pa.types.is_timestamp(schema.field(c).type)]
    id_like = [c for c in names if ID_HINT.search(c)]
    zone_cols = [c for c in names if re.search(r"location_?id", c, re.I)]

    nulls = {c: 0 for c in names}
    geo = {c: dict(n=0, zero=0, in_nyc=0, mn=np.inf, mx=-np.inf) for c in lat_cols + lon_cols}
    dt = {c: dict(mn=None, mx=None) for c in dt_cols}
    per_day = {}                       # pickup date -> count (first datetime column that looks like pickup)
    pickup_col = next((c for c in dt_cols if "pick" in c.lower()), dt_cols[0] if dt_cols else None)
    seen = 0

    for batch in pf.iter_batches(batch_size=a.batch):
        seen += batch.num_rows
        for c in names:
            nulls[c] += batch.column(c).null_count
        for c in lat_cols + lon_cols:
            v = batch.column(c).to_numpy(zero_copy_only=False).astype(float)
            v = v[~np.isnan(v)]
            g = geo[c]
            g["n"] += v.size
            g["zero"] += int((v == 0).sum())
            lo, hi = NYC_LAT if c in lat_cols else NYC_LON
            g["in_nyc"] += int(((v >= lo) & (v <= hi)).sum())
            if v.size:
                g["mn"] = min(g["mn"], float(v.min())); g["mx"] = max(g["mx"], float(v.max()))
        for c in dt_cols:
            v = batch.column(c).to_numpy(zero_copy_only=False)
            v = v[~np.isnat(v)]
            if v.size:
                mn, mx = str(v.min()), str(v.max())
                d = dt[c]
                d["mn"] = mn if d["mn"] is None or mn < d["mn"] else d["mn"]
                d["mx"] = mx if d["mx"] is None or mx > d["mx"] else d["mx"]
        if pickup_col:
            v = batch.column(pickup_col).to_numpy(zero_copy_only=False)
            v = v[~np.isnat(v)].astype("datetime64[D]")
            u, cnt = np.unique(v, return_counts=True)
            for k, n in zip(u, cnt):
                per_day[str(k)] = per_day.get(str(k), 0) + int(n)

    for g in geo.values():
        if g["n"] == 0: g["mn"] = g["mx"] = None

    rep = dict(file=os.path.basename(a.path), rows=n_total, rows_read=seen, columns=cols,
               lat_cols=lat_cols, lon_cols=lon_cols, zone_id_cols=zone_cols,
               id_like_cols=id_like, nulls={k: v for k, v in nulls.items() if v},
               geo=geo, datetime=dt, pickup_col=pickup_col,
               n_days=len(per_day), rows_per_day={k: per_day[k] for k in sorted(per_day)})
    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.path))[0]
    with open(f"results/profile_{stem}.json", "w") as f:
        json.dump(rep, f, indent=1, default=str)

    # small CSV sample
    first = next(pf.iter_batches(batch_size=a.sample)).to_pandas() if a.sample else None
    if first is not None:
        first.to_csv(f"results/sample_{stem}.csv", index=False)

    print("=" * 70); print("FILE:", rep["file"], "| rows:", n_total)
    print("COLUMNS:"); [print("  ", c, t) for c, t in cols]
    print("LAT/LON columns :", lat_cols, lon_cols)
    print("ZONE-ID columns :", zone_cols)
    print("ID-like columns (taxi/driver/vehicle):", id_like or "NONE FOUND")
    for c, g in geo.items():
        print(f"  {c}: n={g['n']} zeros={g['zero']} inside_NYC_box={g['in_nyc']} min={g['mn']} max={g['mx']}")
    for c, d in dt.items(): print(f"  {c}: {d['mn']} -> {d['mx']}")
    print("distinct pickup days:", len(per_day))
    print("NULLS:", rep["nulls"] or "none")
    print("wrote results/profile_%s.json (+ sample csv)" % stem)


if __name__ == "__main__":
    main()
