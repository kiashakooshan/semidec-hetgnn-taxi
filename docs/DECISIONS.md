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

## D9 The 'Chinese' (Baidu) files
CCRNN publishes `taxi_data.h5` / graph files on Baidu. According to both repos' code they hold AGGREGATED counts
(taxi_pick, taxi_drop for 266 virtual stations) and adjacency matrices: no coordinates, no taxi ids. They do not solve the
taxi-level problem (D8). Possible use: validate our pipeline and the CCRNN/DCRNN/Graph WaveNet baselines against the
published numbers. Action: list the files/sizes inside the Baidu share before deciding.
Another route for D8: the paper says its code AND datasets are on GitHub, but the repo contains no data -> ask the authors.

## D10 Third-party mirrors of the CCRNN files (reported by an external assistant, not yet verified here)
Claimed sources: taxi_data.h5 / bike_data.h5 from the ESG repo (Google Drive), all_graph.h5 from Hugging Face
`bjdwh/FlashST-DATA`, adj_mx_*.pkl from `Essaim/CGCDemandPrediction`. We did not receive the files in this project,
and mirrors have no checksum from the original authors. Acceptance test = `scripts/compare_with_ccrnn.py`
(per-slot totals vs our own TLC counts) + `scripts/inspect_all_graph.py`. Only after both pass is the file used.
## D11 Two tracks (proposal)
Track A (the paper's own data): station level on `taxi_data.h5` (266 stations, 4368 x 30 min), features per A-12/A-25,
relations from `all_graph.h5` once their meaning is verified (P18). Track B: zone level from raw TLC (already built).
Neither is taxi level; both are labelled as such.

## Result of the acceptance test for the mirrored taxi_data.h5 (user's run)
Shape (4368, 266) x2; per-slot totals vs our raw-TLC counts: corr 0.9973 at lag 0 (best lag 0) for pick-ups and drop-offs;
kept share of trips 0.881 (their pick-up total 30,735,780 vs our 34,891,690), consistent with CCRNN's 8.42 x 14.45 km box;
their pick-drop gap 3,901 vs ours 5,569. => The file is consistent with the real TLC trips. This is evidence of provenance,
not a byte-identity proof with the Baidu original.
## Meaning of all_graph.h5 matrices (user's run, results/graph_inspect.json)
dis_tt: symmetric, [0,1], diag 1, 27.8% non-zero -> thresholded distance similarity (identical statistics to adj_mx_taxi.pkl[2]).
pcc_tt: correlation with our Pearson matrix of taxi_pick = 1.0000 -> it IS the Pearson correlation of the pick-up series (has negatives, P20).
trans_tt: integer, asymmetric, diag non-zero -> trip counts between stations (direction check pending, scripts/inspect_all_graph.py).
## D12 Two tracks, different jobs
Track A (stations, taxi_data.h5): forecasting accuracy, Table III, baselines.
Track B (TLC zones, real centroids and flows): everything that needs POSITIONS - CLN assignment, boundary nodes, delay (Figs 7-11),
because the 266 stations have no coordinates. Neither is taxi level.
