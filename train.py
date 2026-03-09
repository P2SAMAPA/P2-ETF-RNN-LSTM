#!/usr/bin/env python3
# train.py
# Main training orchestration script — called by GitHub Actions daily
# Implements the full ARMA-RNN-LSTM pipeline from Xiao (2025)
#
# Flow:
#   1. Load OHLC data from HF source dataset
#   2. For each ETF: compute Hurst exponent
#   3. Build feature sequences
#   4. Run 3-stage training (or RNN-only if H ≈ 0.5)
#   5. Generate next-day predictions & rankings
#   6. Save all outputs to HF results dataset

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Setup logging ──────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

import torch
from config import (
    TARGET_ETFS, TRAIN_SPLIT, LOOKBACK, SEED,
    OUT_PREDICTIONS, OUT_RANKINGS, OUT_METRICS, OUT_AUDIT,
)
from data_loader import (
    load_all_data, build_feature_matrix, make_sequences,
    train_test_split_sequences, Normaliser,
)
from hurst import compute_hurst_all_etfs
from trainer import train_pipeline
from hf_io import (
    save_predictions, save_rankings, save_metrics,
    save_audit_trail, save_model_weights, load_model_weights,
    load_existing_results,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    logger.info("=" * 70)
    logger.info("P2-ETF-RNN-LSTM Training Pipeline — Xiao (2025)")
    logger.info(f"Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 70)

    # ── Environment ────────────────────────────────────────────────────────────
    token              = os.environ.get("HF_TOKEN", "")
    retrain_scratch    = os.environ.get("RETRAIN_FROM_SCRATCH", "false").lower() == "true"
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_timestamp      = datetime.now(timezone.utc).isoformat()

    logger.info(f"Device: {device} | Retrain from scratch: {retrain_scratch}")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    logger.info("\n── Step 1: Loading data from HuggingFace ──")
    data = load_all_data(token)

    # ── Step 2: Hurst exponents ────────────────────────────────────────────────
    logger.info("\n── Step 2: Computing Hurst Exponents (R/S Analysis) ──")
    returns_dict = {}
    for etf in TARGET_ETFS:
        if etf in data["ret"].columns:
            ret_series = data["ret"][etf].dropna().values
            returns_dict[etf] = ret_series
        else:
            # Compute from prices if ret not available
            px = data["price"][etf].dropna()
            returns_dict[etf] = np.log(px / px.shift(1)).dropna().values

    hurst_results = compute_hurst_all_etfs(returns_dict)

    # ── Step 3–4: Per-ETF training ────────────────────────────────────────────
    logger.info("\n── Step 3-4: Building features & training models ──")

    all_predictions  = []
    all_rankings_row = []
    all_metrics_rows = []
    all_audit_rows   = []
    today            = data["price"].index[-1]
    pred_date        = _next_trading_day(today)

    for etf in TARGET_ETFS:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing {etf}")

        hurst_info  = hurst_results[etf]
        H           = hurst_info["H"]
        use_hybrid  = hurst_info["use_hybrid"]
        model_label = hurst_info["model"]

        # Build feature matrix
        features = build_feature_matrix(data, etf)
        X, y, dates = make_sequences(features, lookback=LOOKBACK)

        if len(X) < 100:
            logger.warning(f"Not enough data for {etf}, skipping.")
            continue

        # Train/test split
        X_train, y_train, dates_train, X_test, y_test, dates_test = \
            train_test_split_sequences(X, y, dates, split=TRAIN_SPLIT)

        # Normalise (fit on train only — no data leakage, per paper §7.2)
        norm = Normaliser()
        X_train_n = norm.fit_transform(X_train)
        X_test_n  = norm.transform(X_test)

        # Optionally warm-start from saved weights
        # (weights loading handled inside train_pipeline if retrain_scratch=False)

        # Run 3-stage pipeline
        result = train_pipeline(
            X_train=X_train_n,
            y_train=y_train,
            X_test=X_test_n,
            y_test=y_test,
            etf=etf,
            use_hybrid=use_hybrid,
            device=device,
        )

        # Save model weights to HF
        save_model_weights(etf, result, token)

        # ── Generate next-day forecast ────────────────────────────────────────
        # Use the last LOOKBACK rows of the full dataset
        last_features = features.iloc[-LOOKBACK:]
        X_last = norm.transform(
            last_features[[c for c in last_features.columns
                           if c != "target"]].values[np.newaxis, ...]
        )

        if use_hybrid:
            import torch as th
            rnn   = result["rnn"]
            rlstm = result["residual_lstm"]
            hlstm = result["hybrid_lstm"]
            rnn.eval(); rlstm.eval(); hlstm.eval()

            X_t = th.tensor(X_last, dtype=th.float32).to(device)
            with th.no_grad():
                rnn_p   = rnn(X_t).cpu().numpy()
                res_p   = rlstm(X_t).cpu().numpy()
                rnn_col  = np.full((1, X_last.shape[1], 1), rnn_p[0])
                lstm_col = np.full((1, X_last.shape[1], 1), res_p[0])
                X_aug    = np.concatenate([X_last, rnn_col, lstm_col], axis=-1)
                X_aug_t  = th.tensor(X_aug, dtype=th.float32).to(device)
                pred_logret = hlstm(X_aug_t).cpu().numpy()[0]
        else:
            import torch as th
            rnn = result["rnn"]
            rnn.eval()
            X_t = th.tensor(X_last, dtype=th.float32).to(device)
            with th.no_grad():
                pred_logret = rnn(X_t).cpu().numpy()[0]

        current_price = float(data["price"][etf].iloc[-1])
        pred_price    = current_price * np.exp(pred_logret)
        pred_ret_pct  = float(pred_logret * 100)

        metrics = result["metrics"]
        dir_acc = metrics.get("hybrid_dir_acc", metrics.get("rnn_dir_acc", 0.0))

        # ── Collect outputs ───────────────────────────────────────────────────
        all_predictions.append({
            "date":                  pred_date.strftime("%Y-%m-%d"),
            "etf":                   etf,
            "current_price":         round(current_price, 4),
            "predicted_return_pct":  round(pred_ret_pct, 4),
            "predicted_price":       round(float(pred_price), 4),
            "model_used":            model_label,
            "hurst_H":               round(H, 4),
            "memory_type":           hurst_info["memory_type"],
            "direction_accuracy":    round(dir_acc, 2),
            "mae":                   round(metrics.get("hybrid_mae",
                                           metrics.get("rnn_mae", 0)), 6),
            "rmse":                  round(metrics.get("hybrid_rmse",
                                           metrics.get("rnn_rmse", 0)), 6),
            "run_timestamp":         run_timestamp,
        })

        all_metrics_rows.append({
            "run_date":       run_timestamp,
            "etf":            etf,
            "hurst_H":        round(H, 4),
            "memory_type":    hurst_info["memory_type"],
            "model_used":     model_label,
            "train_samples":  len(X_train),
            "test_samples":   len(X_test),
            **{k: round(v, 6) for k, v in metrics.items()},
        })

        all_audit_rows.append({
            "date":             pred_date.strftime("%Y-%m-%d"),
            "signal_etf":       etf,
            "predicted_ret_pct": round(pred_ret_pct, 4),
            "actual_ret_pct":   None,   # filled in next run
            "hurst_H":          round(H, 4),
            "model_used":       model_label,
            "direction_accuracy": round(dir_acc, 2),
            "run_timestamp":    run_timestamp,
        })

    # ── Step 5: Generate rankings ─────────────────────────────────────────────
    logger.info("\n── Step 5: Generating ETF rankings ──")
    sorted_preds = sorted(all_predictions, key=lambda x: x["predicted_return_pct"],
                          reverse=True)
    for rank, pred in enumerate(sorted_preds, 1):
        ret = pred["predicted_return_pct"]
        all_rankings_row.append({
            "date":                 pred_date.strftime("%Y-%m-%d"),
            "rank":                 rank,
            "etf":                  pred["etf"],
            "predicted_return_pct": ret,
            "predicted_price":      pred["predicted_price"],
            "current_price":        pred["current_price"],
            "signal":               "★ BUY" if rank == 1 else ("LONG" if ret > 0 else "AVOID"),
            "model_used":           pred["model_used"],
            "hurst_H":              pred["hurst_H"],
            "direction_accuracy":   pred["direction_accuracy"],
            "run_timestamp":        pred["run_timestamp"],
        })

    top_pick = sorted_preds[0]["etf"] if sorted_preds else "N/A"
    logger.info(f"\n  📊 Rankings for {pred_date.strftime('%Y-%m-%d')}:")
    for r in all_rankings_row:
        logger.info(f"  #{r['rank']} {r['etf']:4s} | "
                    f"{r['predicted_return_pct']:+.3f}% | "
                    f"{r['signal']:8s} | H={r['hurst_H']:.3f} | {r['model_used']}")
    logger.info(f"\n  ★ TOP PICK: {top_pick}")

    # Fill actual returns for yesterday's audit rows (if available)
    _backfill_actual_returns(data, token, run_timestamp)

    # ── Step 6: Save to HF ────────────────────────────────────────────────────
    logger.info("\n── Step 6: Saving outputs to HuggingFace ──")
    save_predictions(all_predictions, token)
    save_rankings(all_rankings_row, token)
    save_metrics(all_metrics_rows, token)
    save_audit_trail(all_audit_rows, token)

    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Pipeline complete! Top pick: {top_pick}")
    logger.info("=" * 70)


def _next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return next weekday after date."""
    nxt = date + timedelta(days=1)
    while nxt.weekday() >= 5:   # skip Sat/Sun
        nxt += timedelta(days=1)
    return nxt


def _backfill_actual_returns(data: dict, token: str, run_timestamp: str):
    """
    Fill in actual_ret_pct for yesterday's audit rows using today's known prices.
    This completes the audit trail with realised performance.
    """
    try:
        existing = load_existing_results("audit_trail.parquet", token)
        if existing is None or len(existing) == 0:
            return

        existing["date"] = pd.to_datetime(existing["date"])
        # Find rows where actual_ret_pct is still null
        mask = existing["actual_ret_pct"].isna()
        if not mask.any():
            return

        # Try to fill from return data
        for idx in existing[mask].index:
            row  = existing.loc[idx]
            etf  = row["signal_etf"]
            date = row["date"]

            if etf in data["ret"].columns and date in data["ret"].index:
                actual = float(data["ret"].loc[date, etf]) * 100
                existing.loc[idx, "actual_ret_pct"] = round(actual, 4)

        from hf_io import _parquet_to_hf, _now_str
        _parquet_to_hf(existing, "audit_trail.parquet", token,
                       f"[auto] Backfill actual returns — {_now_str()}")
        logger.info("Backfilled actual returns in audit trail")

    except Exception as e:
        logger.warning(f"Could not backfill actual returns: {e}")


if __name__ == "__main__":
    main()
