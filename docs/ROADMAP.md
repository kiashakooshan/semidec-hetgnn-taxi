# Roadmap (status after round 8)

| # | Step | Needs | Status |
|---|------|-------|--------|
| 0 | Metrics, Theorem 1 bounds, delay accounting, partition (grid/k-means/k-medoids/FCM), Algorithm 1 (exact), hetGNN-LSTM, FedAvg | - | DONE, 76+ tests |
| 1 | Real data: raw TLC zone counts (Track B), station series + features + splits (Track A) | - | DONE |
| 2 | Meaning of dis_tt / pcc_tt / trans_tt (+ orientation) | - | DONE (P18, P20-P22) |
| 3 | Station coordinates validated | run scripts/validate_station_coordinates.py | WAITING FOR OUTPUT |
| 4 | Station graph builder: A_p, A_c, A_d (A-26), weight cleaning, leak-free variant | decision on A-26 | TODO |
| 5 | Training/eval loop: batches, MSE, Adam (Table II), early stopping, RMSE/MAE/MAPE(%) in counts | GPU for speed | TODO |
| 6 | Baselines DCRNN / Graph WaveNet / CCRNN on the same splits | reference code (GitHub) | TODO |
| 7 | Table III analogue: SC1 full graph, SC2 100 m, SC3 CLN partition | steps 4-5 | TODO |
| 8 | Delay Figs 7-11 with real positions, measured timings, t_cln sweep | step 3 | TODO |
| 9 | Fig 12 analogue (train on subsets of stations, test on all), FL Figs 13-14 | step 5 | TODO |
| 10 | Report: every deviation from the paper (docs/PAPER_ISSUES.md, ASSUMPTIONS.md) | all | TODO |

Not possible with public data: a taxi-level graph (individual taxis with positions/destinations).
