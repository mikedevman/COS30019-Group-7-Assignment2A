import torch
import torch.nn as nn
import copy

# =========================
# GCN LAYER (D^{-1/2} A_tilde D^{-1/2})
# =========================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        # X: (batch, nodes, features)
        # A: (nodes, nodes) — no self-loop in buffer; add I here
        n = A.size(0)
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        A_tilde = A + I
        deg = A_tilde.sum(dim=1).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(deg, -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        out = self.W(A_norm @ X)
        return torch.relu(out)


# =========================
# GCN + LSTM (LSTM on vector state of entire graph at each time step)
# =========================
class GCN_LSTM(nn.Module):
    """
    At each time step: GCN mixes spatial neighbors.
    Time series: LSTM receives (nodes * gcn_hidden) — all nodes after GCN,
    maintains spatial dependencies in time series (does not separate LSTM per station).
    """

    def __init__(
        self,
        A,
        num_nodes,
        in_features,
        hidden_dim=64,
        lstm_hidden=128,
        lstm_num_layers=2,
        dropout_p=0.3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn_hidden = hidden_dim

        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.gcn = GCNLayer(in_features, hidden_dim)

        # Dropout after GCN to reduce overfitting to noise.
        self.dropout_gcn = nn.Dropout(dropout_p)

        # LSTM uses num_layers=2 so dropout inside LSTM is effective.
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=dropout_p if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # Dropout on representation before FC.
        self.dropout_lstm = nn.Dropout(dropout_p)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x, target_idx=None):
        """
        x: (batch, time, nodes, features)
        target_idx:
          - None: return predictions for all nodes => (batch, nodes)
          - int: return prediction for only 1 target node => (batch,)
        """
        batch, time_steps, nodes, _ = x.shape

        # 1) GCN for each time step: (batch, time, nodes, hidden)
        gcn_out = []
        for t in range(time_steps):
            xt = x[:, t, :, :]  # (batch, nodes, features)
            ht = self.gcn(xt, self.A)  # (batch, nodes, hidden)
            ht = self.dropout_gcn(ht)
            gcn_out.append(ht)
        gcn_seq = torch.stack(gcn_out, dim=1)  # (batch, time, nodes, hidden)

        # 2) LSTM per node (each node has time series, but input has been mixed by GCN with k-neighbors)
        #    (batch, nodes, time, hidden) -> (batch*nodes, time, hidden)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).contiguous()
        gcn_flat = gcn_seq.view(batch * nodes, time_steps, self.gcn_hidden)

        lstm_out, _ = self.lstm(gcn_flat)  # (batch*nodes, time, lstm_hidden)
        last_step = lstm_out[:, -1, :]  # (batch*nodes, lstm_hidden)
        last_step = self.dropout_lstm(last_step)

        # 3) Predict for each node then select 1 node if needed
        pred_all = self.fc(last_step).view(batch, nodes)  # (batch, nodes)

        if target_idx is None:
            return pred_all
        return pred_all[:, target_idx]


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0