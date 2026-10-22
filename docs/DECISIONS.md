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
