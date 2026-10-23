# Decisions
## D1 Dataset (proposed, pending TA)
Primary: raw NYC TLC Yellow Taxi, April-June 2016 (the paper's text: 35M trips; ref [13]).
Reason: only raw trips give coordinates for `A_p`, `A_d` and a taxi-level graph, which is the paper's claim.
Gate: after download, check that the files contain pickup/dropoff longitude+latitude. If they contain only zone IDs,
use taxi-zone centroids (TLC taxi-zone shapefile) and document it.
Secondary: CCRNN's `taxi_data.h5` only if the TA has it (region-level baselines). Otherwise region series are
aggregated from the same raw data, so all models share one source.
## D2 Compute
Laptop has Intel Iris Xe (no CUDA). Delay simulation, partitioning, preprocessing: CPU. Model training: Colab/Kaggle GPU or university GPU.

## Update after TA answers
- D1: TA has no dataset -> use raw TLC Yellow Apr-Jun 2016 (official PARQUET). Check with `scripts/profile_tlc.py`.
- D3: Implement the TEXT of the paper, not the authors' notebooks (TA).
- D4: Scope (which figures/tables, FL) decided at the end (TA). Build in this order: data -> graph -> model -> delay -> assignment -> baselines -> FL.
- D5: TA can provide a GPU -> ask for access type (SSH / Jupyter / cluster), GPU model, VRAM.
- D6 (open, meeting with TA): node identity. TA thinks an ID is probably necessary. We will keep trips at EVENT level
  (time, pick-up, drop-off) so that either 'node = trip' or an ID-based definition can be built later.
  The profile script reports whether any taxi/driver/vehicle ID column exists in the real file.

## Update after profiling the real files (3 months, 34,891,690 trips, 91 days)
- Verified: NO lat/lon, NO taxi/driver/vehicle ID. Only zone ids (`PULocationID`, `DOLocationID`).
- D7: Phase 1 builds what the data really supports: pick-up / drop-off counts per zone and 30-min slot
  (`scripts/build_zone_counts.py`) and zone centroids from the official shapefile (`scripts/build_zone_centroids.py`).
  These are needed in every scenario (region-level baselines, Table III region-region, demand/supply features).
- D8 (MEETING WITH TA): the taxi-level graph needs positions and identity. Options:
  (a) accept zone-level resolution: node = trip, position = zone centroid (all trips of a zone share one position;
      100 m radio range and thp/thd thresholds then have coarse meaning);
  (b) find another source that really has coordinates for Apr-Jun 2016 (availability NOT verified);
  (c) a different dataset with taxi ids (then it is no longer the paper's dataset).
  Do not invent positions inside zones (that would be synthetic data).
