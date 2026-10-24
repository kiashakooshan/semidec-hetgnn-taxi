"""hetGNN-LSTM, paper Sec. III-C-1 / Fig. 5 / Table II.

Taxi nodes, three edge types (connectivity, proximity, destination-similarity).  Per layer and relation r a PyG
`GCNConv` (Table II) processes the messages A_r X (Eq. 5-6); the three outputs are combined with the SUM aggregator
(Eq. 7 leaves f unspecified: docs/ASSUMPTIONS.md A-17).  3 hetGNN layers (Table II), then an LSTM (10 layers, Table II).

Architecture choices the paper does not fix (all logged as OPEN in docs/ASSUMPTIONS.md, A-18..A-20):
  * the 216-d node message = P(12) x 18 (9 demand + 9 supply cells) is processed by the hetGNN as ONE vector;
    its 216-d output is reshaped back to a sequence of P steps x 18 and fed to the LSTM; the last LSTM output is mapped
    by a linear layer to the 216-d prediction (= Q x 18 with Q = 12).
  * ReLU between GNN layers, none after the last.  GNN hidden = 216, LSTM hidden = 64 (not given by the paper).
"""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import GCNConv

RELATIONS = ("connectivity", "proximity", "destination")


class HetGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relations=RELATIONS):
        super().__init__()
        self.relations = tuple(relations)
        self.convs = nn.ModuleDict({r: GCNConv(in_dim, out_dim) for r in self.relations})

    def forward(self, x, edge_index: dict, edge_weight: dict | None = None):
        edge_weight = edge_weight or {}
        out = None
        for r in self.relations:
            o = self.convs[r](x, edge_index[r], edge_weight.get(r))
            out = o if out is None else out + o
        return out


class HetGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3, relations=RELATIONS):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers >= 1")
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([HetGCNLayer(dims[i], dims[i + 1], relations) for i in range(n_layers)])

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class HetGNNLSTM(nn.Module):
    def __init__(self, in_dim: int = 216, steps: int = 12, gnn_hidden: int = 216, gnn_layers: int = 3,
                 lstm_hidden: int = 64, lstm_layers: int = 10, out_dim: int = 216, relations=RELATIONS):
        super().__init__()
        if in_dim % steps != 0:
            raise ValueError(f"in_dim {in_dim} must be divisible by steps {steps}")
        self.steps, self.step_dim = steps, in_dim // steps
        self.gnn = HetGNN(in_dim, gnn_hidden, in_dim, gnn_layers, relations)
        self.lstm = nn.LSTM(self.step_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.gnn(x, edge_index, edge_weight)                       # (N, in_dim)
        seq = h.reshape(h.shape[0], self.steps, self.step_dim)         # (N, P, 18)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1])                                   # (N, out_dim)


def make_optimizer(model: HetGNNLSTM) -> torch.optim.Adam:
    """Table II: Adam; GNN lr 0.005 with weight decay 0.001; LSTM lr 0.0001 (LSTM/head: no decay, A-20)."""
    return torch.optim.Adam([
        {"params": model.gnn.parameters(), "lr": 0.005, "weight_decay": 0.001},
        {"params": list(model.lstm.parameters()) + list(model.head.parameters()), "lr": 0.0001, "weight_decay": 0.0},
    ])
