# Assumptions log
Status: OPEN = needs TA/paper confirmation, SET = decided and implemented.

| ID | Assumption | Status |
|----|------------|--------|
| A-01 | Node = a taxi trip that *starts* in the time slot. BLOCKED by data: the real files have NO coordinates and NO taxi ID, only zone ids (see P12). Needs TA meeting. | OPEN |
| A-02 | Demand = pick-ups per cell, supply = drop-offs per cell (as in CCRNN data), because vacant taxis are not observable. | OPEN |
| A-03 | From Table II: Adam, GNN lr 0.005, wd 0.001, L=3, LSTM lr 1e-4, LSTM layers 10, GCNConv, MSE. STILL MISSING: hidden sizes, batch, epochs, time slot, thp, thd, split. | PARTLY SET |
| A-04 | `A_p`, `A_d` weights: paper stores raw distance; we will decide (binary or inverse-distance) after TA input. | OPEN |
| A-05 | Table III reports MAPE in percent -> report with `percent=True`. MAPE ignores entries with y_true = 0 and reports the used fraction (`mape_coverage`). | SET |
| A-06 | Theorem 1: both printed forms (`inside`) and proof form (`outside`) are implemented; N_l = exact-l-hop shell. | SET |
| A-07 | Shared region V_b of a CLN pair = nodes of those two CLNs within `hops` hops of an endpoint of an a-b cross edge (paper says "2-L region", unclear). Default hops=1, configurable. | SET |
| A-08 | Algorithm 2 clustering: 2-means (`n_init=10`, fixed seed) on V_b positions; clusters are mapped to CLNs to minimize moved nodes. | SET |
| A-09 | CLN pairs are visited once in sorted order. | SET |
| A-10 | P = 12 input steps: CONFIRMED by the authors' notebook (12-step blocks, cells 7-8). Q = 12 is only inferred from output dim 216 (the notebook builds input blocks only). | P: SET, Q: OPEN |
| A-11 | TA decision: implement the paper's TEXT, not the authors' notebooks. | SET |
| A-12 | Vicinity of a node = the 9 nodes with the largest weights in its row of `adj_mx_taxi.pkl[2]` (the node itself + 8 nearest); authors' notebook cell 12 (`(-arr).argsort()[:9]`). The weights are similarities in [0,1] with non-zero diagonal, so 'largest' = nearest. | SET (station level) |
| A-13 | Time slot = 30 min. EVIDENCE (user's run of compare_with_ccrnn.py): per-slot totals of the 4368-step file vs our own TLC counts correlate 0.9973 at lag 0 for both pick-ups and drop-offs, and lag 0 is the best of +-4 slots. | SET |
| A-14 | Events outside [2016-04-01, 2016-07-01) or with zone id outside 1..n_zones are excluded and counted in the report; drop-offs earlier than their pick-up are kept and counted in the report. | SET |
| A-15 | n_zones = 265 (TLC ids 1..265; 264/265 are 'Unknown'). CCRNN's 266 nodes are virtual stations (P14), so zone-level and CCRNN-level results are not directly comparable. | OPEN |
| A-16 | Split (CCRNN setup, not stated by this paper): last 672 steps = test (2 weeks), previous 672 = validation, rest = train; see A-28 for the sample rule. | OPEN |
| A-17 | Aggregator f in Eq. (7) = SUM of the three relation outputs (GCNConv per relation, self-loops included by GCNConv). The paper only says 'some aggregation operator'. | OPEN |
| A-18 | Input 216-d vector goes through the hetGNN as one vector; its output is reshaped to (P=12, 18) and fed to the LSTM; last LSTM output -> Linear -> 216-d. Paper does not say how time enters the LSTM. | OPEN |
| A-19 | Hidden sizes not in Table II: GNN hidden = 216, LSTM hidden = 64. ReLU between GNN layers. | OPEN |
| A-20 | Table II 'coefficient decay 0.001' = weight decay of the GNN parameter group only; LSTM/head have none. | OPEN |
| A-21 | Centralized/semi-decentralized upload and download cost ONE ITS-G5 packet time (3.3 ms) each (all taxis transmit concurrently). The paper only says 'time for nodes to upload'. `n_serial_uplinks` makes the alternative explicit. | OPEN |
| A-22 | The decentralized delay is reported as the Theorem-1 bounds, not an 'actual' value: the relaying protocol behind Fig. 7's 'actual' is not specified. t_r must be chosen explicitly (source 23.55 ms vs relay 18.65 ms). | OPEN |
| A-23 | The inter-cloudlet delay t_cln is NOT given by the paper. It is an explicit argument; results are shown as a sensitivity sweep over t_cln, never as a single claimed value. Link is half-duplex, one node-message per packet. | OPEN |
| A-24 | GNN-layer and message-passing-by-computation times are measured on the user's machine with the real model code (reference = 'cloud'); cloudlet x10 and device x100 as stated in the paper. | SET |
| A-25 | Feature layout: ours is time-major, per step [demand of the 9 vicinity nodes, supply of the 9] (18 values) x 12 steps = 216, so that the model's reshape to (12, 18) (A-18) is meaningful. The authors' notebook uses node-major [9 nodes x (12 demand + 12 supply)]; reshaping that to (12, 18) would mix time and space. | SET |
| A-26 | Station-level relations from all_graph.h5 (meaning verified in results/graph_inspect.json): A_p <- dis_tt (distance similarity); A_c <- trans_tt (observed trips between stations) as the stand-in for road connectivity; A_d <- similarity of destination profiles (rows of trans_tt). NOT the paper's definitions (road network, destination coordinates). Proposal, needs TA/user approval. | OPEN |
| A-27 | Normalization: per-channel (pick, drop) z-score with mean/std from TRAIN steps only; metrics are computed after inverse transform, in counts. | SET |
| A-28 | A sample (input P=12 steps, target Q=12 steps) belongs to the split containing its whole TARGET window; inputs may reach into the previous split. | SET |
