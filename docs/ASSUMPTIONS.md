# Assumptions log
Status: OPEN = needs TA/paper confirmation, SET = decided and implemented.

| ID | Assumption | Status |
|----|------------|--------|
| A-01 | Node = a taxi trip that *starts* in the time slot; position = pick-up point, destination = drop-off point (TLC 2016 has no taxi ID). | OPEN |
| A-02 | Demand = pick-ups per cell, supply = drop-offs per cell (as in CCRNN data), because vacant taxis are not observable. | OPEN |
| A-03 | Time-slot length, P, Q, L, hidden size, thp, thd: from paper Table II (user to provide). | OPEN |
| A-04 | `A_p`, `A_d` weights: paper stores raw distance; we will decide (binary or inverse-distance) after TA input. | OPEN |
| A-05 | MAPE ignores entries with y_true = 0 and reports the used fraction (`mape_coverage`). | SET |
| A-06 | Theorem 1: both printed forms (`inside`) and proof form (`outside`) are implemented; N_l = exact-l-hop shell. | SET |
| A-07 | Shared region V_b of a CLN pair = nodes of those two CLNs within `hops` hops of an endpoint of an a-b cross edge (paper says "2-L region", unclear). Default hops=1, configurable. | SET |
| A-08 | Algorithm 2 clustering: 2-means (`n_init=10`, fixed seed) on V_b positions; clusters are mapped to CLNs to minimize moved nodes. | SET |
| A-09 | CLN pairs are visited once in sorted order. | SET |
