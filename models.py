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
    Vanilla RNN — captures SHORT-TERM memory information.
    Input:  (batch, lookback, n_features)
    Output: (batch, hidden_size) — NOTE: Returns hidden state, not final FC prediction yet.
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
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple:
        out, h_n = self.rnn(x, hidden)
        out = self.dropout(out[:, -1, :])
        pred = self.fc(out).squeeze(-1)
        return pred, h_n


class ResidualLSTM(nn.Module):
    """
    LSTM trained on RNN residuals — captures LONG-TERM memory.
    CRITICAL FIX: Input size is now 1 (the scalar residual), not n_features.
    Input:  (batch, lookback, 1)
    Output: (batch, 1)
    """
    def __init__(self, input_size: int = 1, hidden_size: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # Fixed: Expects the 1D residual series
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple:
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])
        pred = self.fc(out).squeeze(-1)
        return pred, (h_n, c_n)


class HybridLSTM(nn.Module):
    """
    Final ARMA-RNN-LSTM Hybrid Model.
    CRITICAL FIX: input_size dynamically handles features + 2 AR components.
    """
    def __init__(self, input_size: int, hidden_size: int = LSTM2_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> tuple:
        out, (h_n, c_n) = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])
        pred = self.fc(out).squeeze(-1)
        return pred, (h_n, c_n)


class ARMARNNLSTMPipeline(nn.Module):
    """
    Full ARMA-RNN-LSTM pipeline.
    CRITICAL FIX: Removed the look-ahead broadcasting bug. 
    The model now outputs raw hidden states so the trainer can correctly 
    construct the autoregressive sequence without temporal leakage.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.rnn = SimpleRNN(input_size)
        self.residual_lstm = ResidualLSTM(input_size=1) # Fixed: Residuals are 1D
        self.hybrid_lstm = HybridLSTM(input_size=input_size + 2) # Features + AR + MA components

    def forward(self, x: torch.Tensor, 
                rnn_h: torch.Tensor = None,
                lstm_h: torch.Tensor = None,
                hybrid_h: torch.Tensor = None) -> dict:
        """
        x: (batch, lookback, n_features)
        Returns a dictionary of predictions and hidden states so the trainer 
        can safely construct the next step without data leakage.
        """
        # Stage 1: Short-term prediction
        rnn_pred, new_rnn_h = self.rnn(x, rnn_h)
        
        # Stage 3 Preparation: We return the raw RNN prediction here.
        # The TRAINER is responsible for creating the residual series and 
        # appending the AR terms to the sequence to avoid look-ahead bias.
        
        return {
            "rnn_pred": rnn_pred,
            "rnn_h": new_rnn_h
        }
