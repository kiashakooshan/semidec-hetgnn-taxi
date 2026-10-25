# semidec-hetgnn-taxi
Re-implementation of *Semi-Decentralized Inference in Heterogeneous GNNs for Traffic Demand Forecasting* (IEEE TVT 2024).

Rule of this project: no synthetic results. Real data, or the paper's own stated formulas/constants. Every gap is in `docs/`.

## Layout
- `src/semidec/metrics.py` RMSE/MAE/MAPE (paper Sec. IV-C)
- `src/semidec/graph/` graph helpers (L-hop shells)
- `src/semidec/delay/` paper constants, Theorem 1 bounds
- `src/semidec/partition/` fixed grid + adaptive node-CLN assignment (Algorithm 2)
- `tests/` hand-computed tests (paper's Fig. 15 graph etc.)
- `docs/` PAPER_ISSUES, ASSUMPTIONS, DECISIONS, TA_QUESTIONS
- `src/semidec/partition/clustering.py` k-medoids and fuzzy c-means (Fig. 11)
- `src/semidec/inference/partitioned.py` Algorithm 1: partitioned inference, proven equal to centralized
- `src/semidec/model/hetgnn_lstm.py` hetGNN-LSTM (Table II); `src/semidec/federated/fedavg.py` server-free FedAvg
- Next phases: `data/` -> preprocess, `graph_builder`, `model`, `delay simulator`, `baselines`, `federated`

## Run
    python -m venv .venv && .venv\Scripts\activate      (Windows)
    pip install -r requirements.txt
    pytest

## Data step (current)
1. Download `yellow_tripdata_2016-04.parquet` (and 05, 06) from the official TLC page into `data/raw/`.
2. `pip install -r requirements.txt`
3. `python scripts/profile_tlc.py data/raw/yellow_tripdata_2016-04.parquet`
4. Send `results/profile_yellow_tripdata_2016-04.json` (and the small sample csv).

## Phase 1 (zone-level data)
    python scripts/build_zone_counts.py          # needs the 3 parquet files in data/raw/
    python scripts/build_zone_centroids.py       # needs data/raw/taxi_zones.zip
Outputs go to `data/processed/` (git-ignored; reproducible).

## Tests
Install: `pip install -r requirements.txt` then `pip install -r requirements-model.txt`.
`pytest` -> torch-dependent tests (`test_model.py`, `test_fedavg.py`) are skipped automatically if torch / torch_geometric are missing.
