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
| A-10 | P = Q = 12 (inferred from 216 = 12 x 2 x 9). The paper never states it, but CCRNN (the source setup) uses P = Q = 12, D = 2. | OPEN (consistent) |
| A-11 | TA decision: implement the paper's TEXT, not the authors' notebooks. | SET |
| A-12 | Paper says "m = n = 9, i.e., a 3x3 grid" (Sec. IV-A): contradictory; 216 = P*2*9 implies m = n = 3 (9 cells). We use 3x3. | OPEN |
| A-13 | Time slot = 30 min. The paper never states it; CCRNN's setup does ('half an hour'; 4368 = 91 x 48). | PROBABLE |
| A-14 | Events outside [2016-04-01, 2016-07-01) or with zone id outside 1..n_zones are excluded and counted in the report; drop-offs earlier than their pick-up are kept and counted in the report. | SET |
| A-15 | n_zones = 265 (TLC ids 1..265; 264/265 are 'Unknown'). CCRNN's 266 nodes are virtual stations (P14), so zone-level and CCRNN-level results are not directly comparable. | OPEN |
| A-16 | Split (CCRNN setup, not stated by this paper): last 4 weeks = 2 weeks validation + 2 weeks test (672 steps each at 30 min); the rest is training. | OPEN |
| A-17 | Aggregator f in Eq. (7) = SUM of the three relation outputs (GCNConv per relation, self-loops included by GCNConv). The paper only says 'some aggregation operator'. | OPEN |
| A-18 | Input 216-d vector goes through the hetGNN as one vector; its output is reshaped to (P=12, 18) and fed to the LSTM; last LSTM output -> Linear -> 216-d. Paper does not say how time enters the LSTM. | OPEN |
| A-19 | Hidden sizes not in Table II: GNN hidden = 216, LSTM hidden = 64. ReLU between GNN layers. | OPEN |
| A-20 | Table II 'coefficient decay 0.001' = weight decay of the GNN parameter group only; LSTM/head have none. | OPEN |
