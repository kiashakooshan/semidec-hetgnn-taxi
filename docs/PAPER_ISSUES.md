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
