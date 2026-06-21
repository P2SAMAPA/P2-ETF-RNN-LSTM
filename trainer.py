# trainer.py
# Sequential 3-stage training for the ARMA-RNN-LSTM pipeline
# Following Xiao (2025) Section 3.2 exactly:
#
#   Stage 1: Train RNN on price/return sequences
#   Stage 2: Compute RNN residuals -> train LSTM on residual sequences
#   Stage 3: Train final HybridLSTM on [original_features | lagged_rnn_pred | lagged_lstm_pred]

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SimpleRNN, ResidualLSTM, HybridLSTM
from config import (
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS_RNN, EPOCHS_LSTM1, EPOCHS_LSTM2,
    BATCH_SIZE, EARLY_STOP_PATIENCE, GRAD_CLIP, SEED, DROPOUT
)

torch.manual_seed(SEED)
logger = logging.getLogger(__name__)


def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _create_lagged_sequences(preds_1d: np.ndarray, lookback: int) -> np.ndarray:
    """
    Create strictly lagged AR terms to prevent look-ahead bias.
    For sample i, the sequence gets preds[i-1] appended. 
    We DO NOT append preds[i] (which is the target being predicted).
    """
    N = len(preds_1d)
    lagged = np.zeros((N, lookback, 1), dtype=np.float32)
    for i in range(lookback, N):
        lagged[i, :, 0] = preds_1d[i - 1]
    return lagged


def _create_residual_sequences(residuals_1d: np.ndarray, lookback: int) -> np.ndarray:
    """
    Build proper rolling windows for the 1D residual series.
    The ResidualLSTM expects input shape (N, lookback, 1).
    """
    N = len(residuals_1d)
    res_seq = np.zeros((N, lookback, 1), dtype=np.float32)
    for i in range(lookback, N):
        res_seq[i, :, 0] = residuals_1d[i - lookback : i]
    return res_seq


def _unpack_pred(pred):
    """Extract the prediction tensor from the (pred, hidden_state) tuple."""
    if isinstance(pred, tuple):
        return pred[0]
    return pred


def _train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    label: str,
    device: torch.device,
) -> list:
    """Generic training loop with early stopping."""
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimiser, patience=5, factor=0.5, verbose=False)
    criterion = nn.MSELoss()
    model.to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            preds = _unpack_pred(model(X_batch)) # CRITICAL FIX: Unpack tuple
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = _unpack_pred(model(X_batch)) # CRITICAL FIX: Unpack tuple
                val_loss += criterion(preds, y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  [{label}] Epoch {epoch:3d}/{epochs} | "
                        f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"  [{label}] Early stop at epoch {epoch} "
                            f"(best val MSE: {best_val_loss:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


def _get_predictions(model: nn.Module, X: np.ndarray,
                     device: torch.device) -> np.ndarray:
    """Run inference over the full array without gradient tracking."""
    model.eval()
    model.to(device)
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = _unpack_pred(model(X_t)) # CRITICAL FIX: Unpack tuple
        return preds.cpu().numpy()


def train_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    etf:     str,
    use_hybrid: bool,
    device:  torch.device = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {etf} | device={device} | hybrid={use_hybrid}")
    logger.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    n_features  = X_train.shape[-1]
    lookback    = X_train.shape[1]
    val_split   = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    train_loader = _make_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    result = {"etf": etf, "use_hybrid": use_hybrid, "stage_histories": {}}

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — SimpleRNN: capture short-term memory
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n  ── Stage 1: SimpleRNN (short-term memory) ──")
    rnn    = SimpleRNN(input_size=n_features)
    hist1  = _train_one_model(rnn, train_loader, val_loader, EPOCHS_RNN,
                               f"{etf}/RNN", device)
    result["rnn"]                       = rnn
    result["stage_histories"]["rnn"]    = hist1

    rnn_train_preds = _get_predictions(rnn, X_train, device)
    rnn_test_preds  = _get_predictions(rnn, X_test,  device)

    if not use_hybrid:
        logger.info(f"  [{etf}] H≈0.5 → using RNN only (no hybrid)")
        result["train_preds"]  = rnn_train_preds
        result["test_preds"]   = rnn_test_preds
        result["final_model"]  = "RNN"
        result["metrics"]      = _compute_metrics(y_test, rnn_test_preds)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — ResidualLSTM: capture long-term memory from RNN residuals
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n  ── Stage 2: ResidualLSTM (long-term memory from residuals) ──")
    rnn_residuals_train = y_train - rnn_train_preds   # ε_t,RNN (1D array)

    # Build rolling windows of the 1D residuals
    res_seq_train = _create_residual_sequences(rnn_residuals_train, lookback)
    res_seq_val   = res_seq_train[val_split:]
    res_seq_tr    = res_seq_train[:val_split]

    if len(res_seq_tr) < 10:
        logger.warning(f"  [{etf}] Not enough residual sequence data. Falling back to RNN.")
        result["train_preds"] = rnn_train_preds
        result["test_preds"]  = rnn_test_preds
        result["final_model"] = "RNN"
        result["metrics"]     = _compute_metrics(y_test, rnn_test_preds)
        return result

    # Targets remain the 1D residuals for the sequence points
    res_targets_train = rnn_residuals_train[lookback:]
    res_targets_val   = rnn_residuals_train[val_split + lookback - 1 : len(rnn_residuals_train)]

    # Ensure lengths match exactly
    min_len = min(len(res_seq_tr), len(res_targets_train))
    res_seq_tr, res_targets_train = res_seq_tr[:min_len], res_targets_train[:min_len]
    min_len_v = min(len(res_seq_val), len(res_targets_val))
    res_seq_val, res_targets_val = res_seq_val[:min_len_v], res_targets_val[:min_len_v]

    res_train_loader = _make_loader(res_seq_tr, res_targets_train, BATCH_SIZE, shuffle=True)
    res_val_loader   = _make_loader(res_seq_val, res_targets_val, BATCH_SIZE, shuffle=False)

    residual_lstm = ResidualLSTM(input_size=1)
    hist2         = _train_one_model(residual_lstm, res_train_loader, res_val_loader,
                                     EPOCHS_LSTM1, f"{etf}/ResidualLSTM", device)
    result["residual_lstm"]                       = residual_lstm
    result["stage_histories"]["residual_lstm"]    = hist2

    lstm_train_preds = _get_predictions(residual_lstm, res_seq_train, device)
    lstm_test_preds  = _get_predictions(residual_lstm, 
                                        _create_residual_sequences(y_test - rnn_test_preds, lookback), 
                                        device)

    combined_train = rnn_train_preds + lstm_train_preds
    combined_test  = rnn_test_preds  + lstm_test_preds

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — HybridLSTM: integrate short + long term for final forecast
    # ══════════════════════════════════════════════════════════════════════════
    logger.info(f"\n  ── Stage 3: HybridLSTM (final integration) ──")

    # Create lagged AR terms (1-step lag, NOT 0-step leak)
    rnn_lagged_train = _create_lagged_sequences(rnn_train_preds, lookback)
    lstm_lagged_train = _create_lagged_sequences(lstm_train_preds, lookback)
    
    rnn_lagged_test = _create_lagged_sequences(rnn_test_preds, lookback)
    lstm_lagged_test = _create_lagged_sequences(lstm_test_preds, lookback)

    # Augment X with lagged predictions
    X_train_aug = np.concatenate([X_tr, rnn_lagged_train[val_split:], lstm_lagged_train[val_split:]], axis=-1).astype(np.float32)
    X_test_aug  = np.concatenate([X_test, rnn_lagged_test, lstm_lagged_test], axis=-1).astype(np.float32)

    # Re-split augmented train into train/val
    val_size_aug = len(X_train_aug) - val_split
    hyb_train_loader = _make_loader(X_train_aug[:val_size_aug],  y_tr,  BATCH_SIZE, shuffle=True)
    hyb_val_loader   = _make_loader(X_train_aug[val_size_aug:], y_val, BATCH_SIZE, shuffle=False)

    hybrid_lstm = HybridLSTM(input_size=n_features + 2)
    hist3       = _train_one_model(hybrid_lstm, hyb_train_loader, hyb_val_loader,
                                   EPOCHS_LSTM2, f"{etf}/HybridLSTM", device)
    result["hybrid_lstm"]                       = hybrid_lstm
    result["stage_histories"]["hybrid_lstm"]    = hist3

    hybrid_train_preds = _get_predictions(hybrid_lstm, X_train_aug, device)
    hybrid_test_preds  = _get_predictions(hybrid_lstm, X_test_aug,  device)

    result["train_preds"]          = hybrid_train_preds
    result["test_preds"]           = hybrid_test_preds
    result["rnn_test_preds"]       = rnn_test_preds
    result["lstm_test_preds"]      = lstm_test_preds
    result["combined_test_preds"]  = combined_test
    result["final_model"]          = "ARMA-RNN-LSTM"
    result["metrics"]              = _compute_metrics(y_test, hybrid_test_preds,
                                                       rnn_test_preds, combined_test)
    logger.info(f"\n  [{etf}] Final metrics: {result['metrics']}")
    return result


def _compute_metrics(y_true, y_hybrid, y_rnn=None, y_combined=None) -> dict:
    """Compute MAE, RMSE, and directional accuracy for each model variant."""

    def mae(a, b):
        min_len = min(len(a), len(b))
        return float(np.mean(np.abs(a[:min_len] - b[:min_len])))

    def rmse(a, b):
        min_len = min(len(a), len(b))
        return float(np.sqrt(np.mean((a[:min_len] - b[:min_len]) ** 2)))

    def dir_acc(a, b):
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        if len(a) < 2:
            return 0.0
        return float(np.mean(np.sign(a[1:]) == np.sign(b[1:])) * 100)

    metrics = {
        "hybrid_mae":     mae(y_true, y_hybrid),
        "hybrid_rmse":    rmse(y_true, y_hybrid),
        "hybrid_dir_acc": dir_acc(y_true, y_hybrid),
    }
    if y_rnn is not None:
        metrics["rnn_mae"]     = mae(y_true, y_rnn)
        metrics["rnn_rmse"]    = rmse(y_true, y_rnn)
        metrics["rnn_dir_acc"] = dir_acc(y_true, y_rnn)
    if y_combined is not None:
        metrics["combined_mae"]     = mae(y_true, y_combined)
        metrics["combined_rmse"]    = rmse(y_true, y_combined)
        metrics["combined_dir_acc"] = dir_acc(y_true, y_combined)
    return metrics
