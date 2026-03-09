# models.py
# PyTorch implementations of:
#   1. SimpleRNN       — captures short-term memory (paper §3, Eq. 24-27)
#   2. ResidualLSTM    — captures long-term memory from RNN residuals (Eq. 28-29)
#   3. HybridLSTM      — final ARMA-RNN-LSTM fusion model (Eq. 30-33)
#
# Reference: Xiao H (2025) PLoS ONE 20(6):e0322737

import torch
import torch.nn as nn
from config import (
    RNN_HIDDEN, LSTM_HIDDEN, LSTM2_HIDDEN,
    RNN_LAYERS, LSTM_LAYERS, DROPOUT, SEED
)

torch.manual_seed(SEED)


class SimpleRNN(nn.Module):
    """
    Vanilla RNN — used to capture SHORT-TERM memory information.

    Per the paper: RNNs are deliberately limited to short-term memory
    due to the vanishing gradient problem. This limitation is EXPLOITED
    in the hybrid model to cleanly isolate short-term dynamics.

    Input:  (batch, lookback, n_features)
    Output: (batch, 1)  — predicted next-day log-return
    """

    def __init__(self, input_size: int, hidden_size: int = RNN_HIDDEN,
                 num_layers: int = RNN_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.rnn(x)
        out    = self.dropout(out[:, -1, :])   # take last timestep
        return self.fc(out).squeeze(-1)


class ResidualLSTM(nn.Module):
    """
    LSTM trained on RNN residuals — captures LONG-TERM memory.

    Per the paper (Eq. 28):
        ε_t,RNN = g_long(p_{t-i}) + ε_t
    The residuals from the RNN contain the long-term memory signal.
    An LSTM is used to extract g_long from those residuals.

    Input:  (batch, lookback, n_features)  — features of the residual series
    Output: (batch, 1)  — predicted residual (= long-term component)
    """

    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm    = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class HybridLSTM(nn.Module):
    """
    Final ARMA-RNN-LSTM Hybrid Model (Eq. 33 in paper).

    Takes as input a concatenation of:
      [original features, rnn_prediction, lstm_residual_prediction]
    and produces the final refined forecast.

    This mirrors the ARMA structure where:
      - RNN  ≡ AR component (historical prices → short-term forecast)
      - LSTM ≡ MA component (error/residual terms → long-term forecast)
      - HybridLSTM ≡ integrator that fuses both

    Input:  (batch, lookback, n_features + 2)
    Output: (batch, 1)
    """

    def __init__(self, input_size: int, hidden_size: int = LSTM2_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm    = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class ARMARNNLSTMPipeline(nn.Module):
    """
    Full ARMA-RNN-LSTM pipeline as a single nn.Module.

    Implements the 3-stage process from Xiao (2025):
      Stage 1: SimpleRNN    → short-term forecast  (p̂_RNN)
      Stage 2: ResidualLSTM → long-term residual   (ε̂_LSTM)
      Stage 3: HybridLSTM   → final forecast       (p̂_hybrid)

    Note: In training, stages are trained sequentially (not end-to-end).
    This module is used for inference only.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.rnn          = SimpleRNN(input_size)
        self.residual_lstm = ResidualLSTM(input_size)
        self.hybrid_lstm  = HybridLSTM(input_size + 2)   # +2 for rnn+lstm preds

    def forward(self, x: torch.Tensor,
                rnn_pred: torch.Tensor,
                lstm_pred: torch.Tensor) -> torch.Tensor:
        """
        x         : (batch, lookback, n_features)
        rnn_pred  : (batch, 1) — appended at each timestep
        lstm_pred : (batch, 1) — appended at each timestep
        """
        # Expand scalar preds to (batch, lookback, 1) and concatenate
        rnn_exp  = rnn_pred.unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), 1)
        lstm_exp = lstm_pred.unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), 1)
        x_aug    = torch.cat([x, rnn_exp, lstm_exp], dim=-1)
        return self.hybrid_lstm(x_aug)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
