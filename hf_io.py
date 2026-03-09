# hf_io.py
# HuggingFace dataset I/O — reads source data, writes results
# Source:  P2SAMAPA/p2-etf-deepwave-dl
# Results: P2SAMAPA/p2-etf-rnn-lstm-results

import os
import io
import json
import logging
import tempfile
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, upload_file
from config import HF_RESULTS_DATASET, OUT_WEIGHTS_DIR

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Add it as a GitHub Actions secret or .env file."
        )
    return token


def _parquet_to_hf(df: pd.DataFrame, filename: str, token: str,
                   commit_message: str = None):
    """Upload a DataFrame as parquet to the results HF dataset."""
    api = HfApi()
    buf = io.BytesIO()
    df.to_parquet(buf, index=True, engine="pyarrow")
    buf.seek(0)

    msg = commit_message or f"[auto] Update {filename} — {_now_str()}"
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=filename,
        repo_id=HF_RESULTS_DATASET,
        repo_type="dataset",
        token=token,
        commit_message=msg,
    )
    logger.info(f"Uploaded {filename} → {HF_RESULTS_DATASET}")


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Read from results dataset ─────────────────────────────────────────────────

def load_existing_results(filename: str, token: str = None) -> pd.DataFrame | None:
    """
    Try to load an existing parquet file from the results dataset.
    Returns None if file doesn't exist yet.
    """
    try:
        token = token or _get_token()
        local = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        df = pd.read_parquet(local)
        logger.info(f"Loaded existing {filename}: {len(df)} rows")
        return df
    except Exception as e:
        logger.info(f"No existing {filename} found ({e}) — will create fresh.")
        return None


# ── Write predictions ─────────────────────────────────────────────────────────

def save_predictions(predictions: list[dict], token: str = None):
    """
    Save/append daily predictions to predictions.parquet.

    Each dict in predictions should have:
      date, etf, current_price, predicted_return_pct,
      predicted_price, model_used, hurst_H, direction_accuracy,
      mae, rmse, run_timestamp
    """
    token = token or _get_token()
    new_df = pd.DataFrame(predictions)
    new_df["date"] = pd.to_datetime(new_df["date"])

    existing = load_existing_results("predictions.parquet", token)
    if existing is not None:
        existing["date"] = pd.to_datetime(existing["date"])
        # Remove any rows for today's date to avoid duplicates
        today = new_df["date"].max()
        existing = existing[existing["date"] < today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["date", "etf"]).reset_index(drop=True)
    _parquet_to_hf(combined, "predictions.parquet", token,
                   f"[auto] Predictions update — {_now_str()}")
    return combined


def save_rankings(rankings: list[dict], token: str = None):
    """
    Save/append daily ETF rankings to rankings.parquet.

    Each dict: date, rank, etf, predicted_return_pct,
                signal (BUY/LONG/AVOID), model_used, hurst_H
    """
    token = token or _get_token()
    new_df = pd.DataFrame(rankings)
    new_df["date"] = pd.to_datetime(new_df["date"])

    existing = load_existing_results("rankings.parquet", token)
    if existing is not None:
        existing["date"] = pd.to_datetime(existing["date"])
        today = new_df["date"].max()
        existing = existing[existing["date"] < today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["date", "rank"]).reset_index(drop=True)
    _parquet_to_hf(combined, "rankings.parquet", token,
                   f"[auto] Rankings update — {_now_str()}")
    return combined


def save_metrics(metrics_rows: list[dict], token: str = None):
    """Save/append per-ETF training metrics to metrics.parquet."""
    token = token or _get_token()
    new_df = pd.DataFrame(metrics_rows)
    new_df["run_date"] = pd.to_datetime(new_df["run_date"])

    existing = load_existing_results("metrics.parquet", token)
    if existing is not None:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["run_date", "etf"]).reset_index(drop=True)
    _parquet_to_hf(combined, "metrics.parquet", token,
                   f"[auto] Metrics update — {_now_str()}")
    return combined


def save_audit_trail(audit_rows: list[dict], token: str = None):
    """
    Save/append audit trail rows to audit_trail.parquet.

    Each row: date, signal_etf, rank_1_etf, predicted_ret_pct,
              actual_ret_pct (filled next day), all ETF actual returns,
              regime_label, hurst_H, model_used, conviction_score
    """
    token = token or _get_token()
    new_df = pd.DataFrame(audit_rows)
    new_df["date"] = pd.to_datetime(new_df["date"])

    existing = load_existing_results("audit_trail.parquet", token)
    if existing is not None:
        existing["date"] = pd.to_datetime(existing["date"])
        # Update existing rows (fills in actual_ret once known)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "signal_etf"],
                                             keep="last")
    else:
        combined = new_df

    combined = combined.sort_values("date").reset_index(drop=True)
    _parquet_to_hf(combined, "audit_trail.parquet", token,
                   f"[auto] Audit trail update — {_now_str()}")
    return combined


# ── Model weights ─────────────────────────────────────────────────────────────

def save_model_weights(etf: str, pipeline_result: dict, token: str = None):
    """
    Save PyTorch model state dicts to HF results dataset.
    Saves: weights/{etf}_rnn.pt, weights/{etf}_residual_lstm.pt,
           weights/{etf}_hybrid_lstm.pt (if hybrid)
    """
    token = token or _get_token()
    api   = HfApi()

    models_to_save = {"rnn": pipeline_result.get("rnn")}
    if pipeline_result.get("use_hybrid"):
        models_to_save["residual_lstm"] = pipeline_result.get("residual_lstm")
        models_to_save["hybrid_lstm"]   = pipeline_result.get("hybrid_lstm")

    for name, model in models_to_save.items():
        if model is None:
            continue
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        path_in_repo = f"{OUT_WEIGHTS_DIR}/{etf}_{name}.pt"
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=HF_RESULTS_DATASET,
            repo_type="dataset",
            token=token,
            commit_message=f"[auto] Weights {etf}/{name} — {_now_str()}",
        )
        logger.info(f"Saved weights → {path_in_repo}")


def load_model_weights(etf: str, model, name: str, token: str = None) -> bool:
    """
    Load saved weights into model. Returns True if successful.
    Used for warm-starting training runs.
    """
    try:
        token = token or _get_token()
        local = hf_hub_download(
            repo_id=HF_RESULTS_DATASET,
            filename=f"{OUT_WEIGHTS_DIR}/{etf}_{name}.pt",
            repo_type="dataset",
            token=token,
        )
        state = torch.load(local, map_location="cpu")
        model.load_state_dict(state)
        logger.info(f"Loaded warm-start weights for {etf}/{name}")
        return True
    except Exception as e:
        logger.info(f"No saved weights for {etf}/{name}: {e}")
        return False
