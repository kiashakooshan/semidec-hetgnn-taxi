# Hyperparameters  (source: paper Table II, read from the user's screenshot)

| Property | Value | Status |
|---|---|---|
| Optimizer | Adam | given |
| GNN learning rate | 0.005 | given |
| GNN coefficient decay (weight decay) | 0.001 | given |
| GNN no. of layers | 3 | given  (L = 3) |
| LSTM learning rate | 0.0001 | given |
| LSTM no. of layers | 10 | given |
| Input dimensions | 216 | given |
| Output dimensions | 216 | given |
| Loss | MSE | given |
| GNN convolution layer | `GCNConv` from PyG | given |

## Inference (NOT stated in the paper -> must be confirmed)
216 = 12 x 2 x 9  = P x (demand, supply) x (3x3 vicinity)  =>  P = Q = 12.
This matches CCRNN's 12-in / 12-out setting, but the paper never says it. (A-10)

## Still missing from the paper
hidden sizes (GNN and LSTM), batch size, number of epochs, time-slot length, thp, thd, train/val/test split,
normalization. They must be chosen by us and logged in `docs/ASSUMPTIONS.md`, not hidden in code.
