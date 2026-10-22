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
- Next phases: `data/` -> preprocess, `graph_builder`, `model`, `delay simulator`, `baselines`, `federated`

## Run
    python -m venv .venv && .venv\Scripts\activate      (Windows)
    pip install -r requirements.txt
    pytest
