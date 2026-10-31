# Issues found in the paper and in the authors' public code

Everything below was verified by reading the PDF and the repo
`mahmoudkanazzal/SemidecentralizedhetGNNLSTM` (2 commits; 9 notebooks + 5 MATLAB files; no .py, no data, no README).
Each item says *where* so it can be re-checked.

## A. Paper (Theorem 1 and Appendix)
| ID | Issue | Evidence |
|----|-------|----------|
| P1 | Eq. (8) and Eq. (16) put `(l+1)t_p` **inside** `2 t_r [...]`; Eq. (14)/(15) put it **outside**. They are different formulas. | Eq. (8) vs (15) |
| P2 | `N_l(i)` is defined as nodes "l hops away", but the 2-hop proof uses the degrees of the 1-hop relays j,k. On the paper's own Fig. 15 graph: proof reading gives lower=18.3, shell reading gives 10.3 (t_r=1, t_p=0.1). | `tests/test_theorem1.py` |
| P3 | Text claims delay grows **quadratically** in L, but Eq. (8) contains `sum_l (L-l+1)*l*d_i = d_i*L(L+1)(L+2)/6`, which is **cubic** in L. | `test_printed_weight_sum_is_cubic_not_quadratic` |
| P4 | Eq. (11): `2(ts+tr)+tp = 2x2(tr+tp)` is false unless tp=0; Eq. (12) `2 d_i (tr+tp)` conflicts with the derivation that counts tp once per layer. | Appendix |
| P5 | Typos: Eq. (3) `dist(di,d2)` (should be dj); Eq. (7) `M^a` (should be `M^p`); Sec. IV-C says "MSE" but tables report RMSE; Fig. 10 says "CLA" (means CLN); Sec. IV-C "S2" (means SC2). | text |
| P6 | Eq. (2)/(3) store the raw *distance* as the edge weight (larger distance = larger weight). Usually a similarity (inverse/kernel) is intended. | Eq. (2),(3) |
| P7 | Adaptive assignment uses k-means on node **positions** only; it never looks at edges, so it cannot guarantee fewer cross-CLN edges (it is a heuristic, not min-cut). | Sec. III-C-2 |
| P8 | Table II gives optimizer/lr/layers/dims but NOT hidden sizes, batch, epochs, P, Q, slot length, thp, thd. | Table II |
| P12 | Sec. IV-A says each trip has pick-up/drop-off longitude and latitude, and the method needs coordinates (A_p, A_d, 100 m range). The official Apr-Jun 2016 PARQUET release has only `PULocationID`/`DOLocationID`. Verified by profiling all three files (34,891,690 rows, no lat/lon, no ID-like column). The row total does match the paper's '35 million trips'. | results/profile_*.json |
| P13 | The dataset sentence of Sec. IV-A (trip fields incl. pick-up/drop-off longitude and latitude, 35M trips, Apr-Jun 2016) has the same content as the CCRNN paper's description of NYC Taxi. Table III's DCRNN (14.79 / 8.43) and Graph WaveNet (13.07 / 8.10) RMSE/MAE equal, to 2 decimals, the numbers PUBLISHED in CCRNN's Table 1 (14.7926 / 8.4274 and 13.0729 / 8.1037), while CCRNN differs (9.69 / 5.58 here vs 9.5631 / 5.4979 published). Consistent with those two baselines being copied rather than re-run; not proof. | CCRNN Table 1 |
| P14 | CCRNN's 266 nodes are 'virtual stations' from density-peak clustering of pick-up COORDINATES inside an 8.42 km x 14.45 km box (West NYC, Manhattan, part of Brooklyn); they are NOT TLC zones and cannot be rebuilt from zone-level files. | CCRNN, Experimental Setup |
| P15 | Fig. 11 claims the clustering criterion has only a marginal effect. On a small hand-built two-blob graph, k-means and FCM keep the cut at 2 edges, but k-medoids with the Park-Jun initialisation (ref [60]) splits along the wrong axis and raises it to 6 (`tests/test_clustering.py`). One toy example is not a refutation, but the claim is not guaranteed; real graphs decide. | tests |
| P16 | Magnitudes in Fig. 7/8 (read by eye): centralized delay at L=1 is ~2-3 ms and decentralized ~2 ms, yet the text says the centralized time includes two ITS-G5 transmissions (2 x 3.3 = 6.6 ms) and the decentralized one at least one ad-hoc transmission (18.65-23.55 ms). The plotted values cannot contain the stated constants in ms; so the figures cannot be reproduced numerically, only their trends. | Fig. 7/8 vs Sec. IV-D |
| P17 | Complexity claim: cross-CLN cost 'gamma N/K L^2 t_CLN'. Our exact Algorithm 1 (tests/test_partitioned.py) needs only the 1-hop halo per layer: messages per layer are constant, total LINEAR in L, and the result equals centralized exactly. The L^2 term belongs to a protocol that relays L-hop messages. | tests |
| P18 | The authors' own notebook (cell 1) asks 'what is meant by trans_tt, pcc_tt, and dis_tt?': they did not know what the three matrices of `all_graph.h5` mean, yet use them (R3: one matrix for all relations). The names suggest distance / Pearson correlation / transition; `scripts/inspect_all_graph.py` tests this against data. | authors' nb cell 1 |
| P19 | Authors' feature layout (node-major, 9 x (12 dem + 12 sup)) differs from a time-major (12 x 18) layout; the paper's Eq. (4) uses time-major tensors X in R^{P x N x d}. We follow the paper (A-25). | authors' nb cells 8-12 |
| P20 | `pcc_tt` has NEGATIVE entries (min -0.46, row sums down to -23.6). Used raw as GCN edge weights, the weighted degree can be <= 0 and D^-1/2 gives NaN. It needs clipping / thresholding first. | graph_inspect.json |
| P21 | `pcc_tt` is the Pearson correlation of the pick-up series (correlation with our own Pearson matrix = 1.0000) and `trans_tt` is an integer trip-count matrix; both were computed over the WHOLE 91 days, i.e. including the test weeks. Using them as the graph for forecasting the test period leaks future information. `pcc_tt` can be recomputed from the training steps; `trans_tt` cannot (no station-level trips). | graph_inspect.json |
| P22 | trans_tt is [destination, origin], not [origin, destination] (direction check: corr(row sums, drop-offs) = 1.0000 vs corr(row sums, pick-ups) = 0.9619; column sums the other way round). Read naively it would flip every flow. | graph_inspect.json |
| P23 | SC2 uses a 100 m ad-hoc range. With the recovered station positions only 2 of 266 stations have ANY neighbour within 100 m (mean 0.01 neighbours; 159 within 200 m, 256 within 500 m). At station level SC2 is therefore an almost edgeless graph; the paper's SC2 numbers cannot come from a 100 m radius on these nodes. | station_coords CSV |
| P9 | Sec. IV-A says "m = n = 9, i.e., a 3x3 grid"; 9x9 and 3x3 are different. Input dim 216 fits 3x3 (12*2*9). | Sec. IV-A, Table II |
| P10 | Sec. IV-C says reductions of 45.4% / 56.0% / 58.5% in "MSE, MAE, MAPE". Computed from Table III (CCRNN 9.69 -> 5.29) 45.4% is the **RMSE** reduction, not MSE (MSE would be ~70%). Arithmetic of the percentages is otherwise consistent with the table. | Table III |
| P11 | Table III: the 'Proposed' rows are identical in the Taxi-region and Taxi-taxi cases (same 14.17 / 11.99 / 10.24 ...), although the text says taxi-region compares different quantities. | Table III |

## B. Authors' code
| ID | Finding | Evidence |
|----|---------|----------|
| R1 | Repo has **no data and no .py files**; datasets are `taxi_data.h5` (from Baidu, CCRNN) and `all_graph.h5` (source unknown). | repo tree, FL notebook cell 1 |
| R2 | "Taxis" are **not real taxis**: nodes come from CCRNN's 266 *regions*; per-region feature vectors are replicated with `np.matlib.repmat` n times (n = taxis in region), so taxis of one region have identical features. Only the first 40 regions are used (`xx=40`). | PE/FL/Generalization nb: `get_graph_per_ts` |
| R3 | The three edge types are built from the **same** matrix: `A0`, `A1`, `A2` all use `adj_0[i,i]` (block-diagonal constant blocks). So the graph is effectively homogeneous. | same function |
| R4 | The result notebooks train a `HeteroGNN` only (GCN/SAGE/GAT convs, hidden 300, 3 layers). No LSTM is trained; the LSTM class exists only in `Demo_feb_12.ipynb`. | notebooks |
| R5 | Fig. 12 and Fig. 14 values are **typed constants** (`rmse_arr=np.array([...])`) followed by `np.sort(...)[::-1]`, i.e. sorted descending before plotting. As typed: FL RMSE and MAE, and Generalization MAE, are *not* monotone; after sorting they are. No code in the repo computes these numbers. | FL nb cells 55-57; Generalization nb cells 55-58 |
| R6 | FL notebook: 5 workers x 9 local epochs x 1 aggregation (paper: 10 clients, 10 rounds, 50 epochs). Recorded outputs there are normalized RMSE ~0.97-0.99, while Fig. 14 shows ~15. Plot legend says "94 adversary nodes" (copied from another project). | FL nb cells 30-55 |
| R7 | MATLAB delay code (Figs. 7-11) uses random node positions and hand-set constants (e.g. `proc_delay=0.1, 0.001, 0.005`, `*10^-1`). The paper's constants (3.3 ms, 16.55+7, 11.65+7) are absent from the main file; no code for Theorem 1 bounds exists. | `Demo_1_4_scenarios_delay_overhead*.m` |
| R9 | Table III 'Proposed-SC1' RMSE 14.17 and MAE 11.99 equal (to 2 decimals) the LAST typed constants of the Fig. 12 arrays in the Generalization notebook (14.1676 and 11.9946). This is consistent with Table III values coming from those typed constants rather than a new run; it is not proof. | Generalization nb vs Table III |
| R8 | `Demo_jan_12/feb_14` are runs of the **CCRNN** baseline (batch 32, 12-in/12-out, 266 nodes, 2 features = pick-up, drop-off). So "supply" in the authors' data = drop-offs. | CCRNN `data_container.py` |

## Consequence for this project
The authors' code is used only as a *reference for data formats*. All results here are produced by our own code
from real data or from the paper's own stated formulas/constants; discrepancies are reported, not hidden.
