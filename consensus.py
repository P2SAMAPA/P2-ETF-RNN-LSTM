# consensus.py
# Consensus Sweep — runs inference across all training-start years in parallel
# and combines signals using a weighted conviction score.
#
# Conviction formula (v2):
#   score = (vote_share × 0.35) + (norm_dir_acc × 0.40) + (norm_avg_H × 0.25)
#
# Usage (called from app.py):
#   from consensus import run_consensus_sweep, save_consensus_results, load_consensus_results

import os
import io
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from config
from config import TARGET_ETFS, ETF_LABELS

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONSENSUS_YEARS     = list(range(2008, 2025))   # 2008 → 2024 inclusive
CONSENSUS_FILE      = "consensus/consensus_latest.parquet"
CONSENSUS_DIR       = "consensus"
TRAIN_SPLIT         = 0.80


# ── Helper ─────────────────────────────────────────────────────────────────────
def _next_trading_day(from_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekday (Mon–Fri) after from_date."""
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ── Per-year inference (runs in thread) ───────────────────────────────────────

def _infer_one_year(year: int, price_df: pd.DataFrame,
                    ret_df: pd.DataFrame) -> dict | None:
    """
    Run the full ARMA-RNN-LSTM pipeline for a single training-start year.
    Returns a dict of per-ETF results, or None on failure.
    """
    try:
        import torch
        from data_loader import (build_feature_matrix, make_sequences,
                                  train_test_split_sequences, Normaliser)
        from hurst import hurst_exponent, classify_memory
        from trainer import train_pipeline

        cutoff  = pd.Timestamp(f"{year}-01-01")
        pf      = price_df[price_df.index >= cutoff]
        rf      = ret_df[ret_df.index >= cutoff] if ret_df is not None else None

        if len(pf) < 200:
            return None

        device  = torch.device("cpu")
        results = {}

        for etf in TARGET_ETFS:
            if etf not in pf.columns:
                continue
            try:
                data_full = {
                    "price":       pf,
                    "ret":         rf if rf is not None else pd.DataFrame(),
                    "vol":         pd.DataFrame(index=pf.index),
                    "bench_price": pd.DataFrame(index=pf.index),
                    "bench_ret":   pd.DataFrame(index=pf.index),
                    "bench_vol":   pd.DataFrame(index=pf.index),
                }
                ret_s = (rf[etf].dropna().values
                         if rf is not None and etf in rf.columns
                         else np.diff(np.log(pf[etf].dropna().values)))

                H   = hurst_exponent(ret_s)
                mem = classify_memory(H)

                features       = build_feature_matrix(data_full, etf)
                X, y, dates    = make_sequences(features)
                if len(X) < 50:
                    continue

                X_tr, y_tr, _, X_te, y_te, _ = train_test_split_sequences(
                    X, y, dates, TRAIN_SPLIT)

                norm = Normaliser()
                res  = train_pipeline(norm.fit_transform(X_tr), y_tr,
                                      norm.transform(X_te),     y_te,
                                      f"{etf}_{year}", mem["use_hybrid"], device)

                current_price = float(pf[etf].iloc[-1])
                pred_logret   = float(res["test_preds"][-1]) if len(res["test_preds"]) else 0.0
                dir_acc       = res["metrics"].get(
                    "hybrid_dir_acc", res["metrics"].get("rnn_dir_acc", 50.0))

                results[etf] = {
                    "year":            year,
                    "etf":             etf,
                    "H":               H,
                    "model":           mem["model"],
                    "pred_ret_pct":    pred_logret * 100,
                    "predicted_price": current_price * np.exp(pred_logret),
                    "current_price":   current_price,
                    "dir_acc":         dir_acc,
                }
            except Exception as e:
                logger.warning(f"[consensus] {etf}/{year} failed: {e}")
                continue

        return results if results else None

    except Exception as e:
        logger.error(f"[consensus] Year {year} outer failure: {e}")
        return None


# ── Conviction scorer ──────────────────────────────────────────────────────────

def _compute_conviction(all_year_results: list[dict]) -> pd.DataFrame:
    """
    Given a flat list of {year, etf, pred_ret_pct, H, dir_acc, ...} dicts,
    compute conviction score per ETF.

    Score (v2) = 0.35 × vote_share
               + 0.40 × norm_dir_acc   (min-max normalised across ETFs)
               + 0.25 × norm_avg_H     (min-max normalised)

    avg_pred_ret is stored for display only — not used in scoring.
    """
    df = pd.DataFrame(all_year_results)
    if df.empty:
        return pd.DataFrame()

    # Which ETF ranked #1 per year by predicted return?
    top_per_year = (df.sort_values("pred_ret_pct", ascending=False)
                      .groupby("year")
                      .first()
                      .reset_index()[["year", "etf"]])
    vote_counts  = top_per_year["etf"].value_counts()
    total_years  = len(top_per_year["year"].unique())

    # Aggregate per ETF
    agg = (df.groupby("etf")
             .agg(avg_pred_ret=("pred_ret_pct", "mean"),
                  avg_H=("H",            "mean"),
                  avg_dir_acc=("dir_acc", "mean"),
                  year_count=("year",     "count"))
             .reset_index())

    agg["votes"]      = agg["etf"].map(vote_counts).fillna(0).astype(int)
    agg["vote_share"] = agg["votes"] / total_years

    # Min-max normalise dir_acc and H (handle edge case: all equal)
    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 1e-9 else pd.Series([0.5] * len(s), index=s.index)

    agg["norm_dir_acc"] = minmax(agg["avg_dir_acc"])
    agg["norm_H"]       = minmax(agg["avg_H"])
    # Keep norm_ret for backwards compat but don't use in score
    agg["norm_ret"]     = minmax(agg["avg_pred_ret"])

    # v2 conviction formula: votes 35%, dir_acc 40%, H 25%
    agg["conviction"] = (0.35 * agg["vote_share"]
                       + 0.40 * agg["norm_dir_acc"]
                       + 0.25 * agg["norm_H"])

    agg["rank"] = agg["conviction"].rank(ascending=False, method="first").astype(int)
    agg = agg.sort_values("rank").reset_index(drop=True)

    agg["label"]    = agg["etf"].map(ETF_LABELS).fillna("")
    agg["run_date"] = pd.Timestamp(datetime.now(timezone.utc).date())

    return agg


# ── Main sweep entry point ─────────────────────────────────────────────────────

def run_consensus_sweep(price_df: pd.DataFrame,
                        ret_df: pd.DataFrame,
                        years: list[int] = None,
                        max_workers: int = 4,
                        progress_callback=None) -> dict:
    """
    Run all-years consensus sweep in parallel threads.

    Returns
    -------
    dict with keys:
        "conviction"     : pd.DataFrame  (ranked ETFs with conviction scores)
        "all_results"    : list[dict]     (flat per-year per-ETF rows)
        "year_tops"      : dict           {year: top_etf}
        "years_run"      : int
        "years_failed"   : int
        "run_ts"         : str (ISO UTC timestamp of when sweep ran)
        "signal_date"    : str (next trading day — the date being predicted for)
    """
    if years is None:
        years = CONSENSUS_YEARS

    all_flat  = []
    year_tops = {}
    failed    = 0
    done      = 0
    total     = len(years)

    if progress_callback:
        progress_callback(0.0, f"Starting sweep across {total} training windows…")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_infer_one_year, yr, price_df, ret_df): yr
                   for yr in years}

        for fut in as_completed(futures):
            yr = futures[fut]
            done += 1
            pct  = done / total

            try:
                res = fut.result()
                if res:
                    for etf, row in res.items():
                        all_flat.append(row)
                    top = max(res.values(), key=lambda r: r["pred_ret_pct"])
                    year_tops[yr] = top["etf"]
                    status = f"✅ {yr} → top: {top['etf']} ({top['pred_ret_pct']:+.3f}%)"
                else:
                    failed += 1
                    status = f"⚠️ {yr} → no results (insufficient data)"
            except Exception as e:
                failed += 1
                status = f"❌ {yr} → error: {e}"

            if progress_callback:
                progress_callback(pct, f"[{done}/{total}] {status}")

    conviction_df = _compute_conviction(all_flat)

    # run_ts = when the sweep actually ran (UTC)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # signal_date = next trading day after today (what we're predicting for)
    _today = pd.Timestamp(datetime.now(timezone.utc).date())
    signal_date = _next_trading_day(_today).strftime("%Y-%m-%d")

    if progress_callback:
        top_etf = conviction_df.iloc[0]["etf"] if len(conviction_df) > 0 else "—"
        progress_callback(1.0, f"✅ Sweep complete — consensus pick: **{top_etf}** for {signal_date}")

    return {
        "conviction":   conviction_df,
        "all_results":  all_flat,
        "year_tops":    year_tops,
        "years_run":    done - failed,
        "years_failed": failed,
        "run_ts":       run_ts,
        "signal_date":  signal_date,
    }


# ── HuggingFace persistence ────────────────────────────────────────────────────

def save_consensus_results(sweep_result: dict, token: str = None) -> bool:
    """
    Push consensus results to HF results dataset.
    Keeps ONLY the latest run (date-stamped filename + pointer file).
    """
    try:
        from huggingface_hub import HfApi
        from config import HF_RESULTS_DATASET

        token = token or os.environ.get("HF_TOKEN", "")
        if not token:
            logger.warning("HF_TOKEN not set — consensus results not saved to HF")
            return False

        api = HfApi()
        df  = sweep_result["conviction"].copy()
        df["run_ts"]     = sweep_result["run_ts"]
        df["years_run"]  = sweep_result["years_run"]
        df["signal_date"] = sweep_result.get("signal_date", "")

        flat_df = pd.DataFrame(sweep_result["all_results"])
        flat_df["run_ts"]     = sweep_result["run_ts"]
        flat_df["signal_date"] = sweep_result.get("signal_date", "")

        run_ts_clean = sweep_result["run_ts"][:16].replace(":", "").replace("-", "").replace("T", "_")
        stamped_name = f"{CONSENSUS_DIR}/consensus_{run_ts_clean}.parquet"

        def _upload(df_to_upload, path):
            buf = io.BytesIO()
            df_to_upload.to_parquet(buf, index=False, engine="pyarrow")
            buf.seek(0)
            api.upload_file(
                path_or_fileobj=buf,
                path_in_repo=path,
                repo_id=HF_RESULTS_DATASET,
                repo_type="dataset",
                token=token,
                commit_message=f"[auto] Consensus update — {sweep_result['run_ts']}",
            )

        _upload(df, stamped_name)
        _upload(df, CONSENSUS_FILE)

        flat_stamped = f"{CONSENSUS_DIR}/flat_{run_ts_clean}.parquet"
        _upload(flat_df, flat_stamped)
        _upload(flat_df, f"{CONSENSUS_DIR}/flat_latest.parquet")

        # ── Cleanup old stamped files ──────────────────────────────────────
        try:
            from huggingface_hub import list_repo_files
            all_files = list(list_repo_files(HF_RESULTS_DATASET,
                                             repo_type="dataset", token=token))
            to_delete_stamped = sorted([
                f for f in all_files
                if f.startswith(f"{CONSENSUS_DIR}/consensus_2") and f != stamped_name
            ])
            to_delete_flat = sorted([
                f for f in all_files
                if f.startswith(f"{CONSENSUS_DIR}/flat_2") and f != flat_stamped
            ])
            # Keep only the most recent previous stamped file (i.e. delete all but last 1)
            for old_file in to_delete_stamped[:-1] + to_delete_flat[:-1]:
                try:
                    api.delete_file(path_in_repo=old_file,
                                    repo_id=HF_RESULTS_DATASET,
                                    repo_type="dataset", token=token,
                                    commit_message=f"[auto] Cleanup — {old_file}")
                    logger.info(f"Deleted old consensus file: {old_file}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Cleanup step failed (non-critical): {e}")

        logger.info(f"Consensus saved → {stamped_name} + {CONSENSUS_FILE}")
        return True

    except Exception as e:
        logger.error(f"Failed to save consensus to HF: {e}")
        return False


def load_consensus_results(token: str = None) -> tuple[pd.DataFrame | None,
                                                        pd.DataFrame | None]:
    """
    Load the latest consensus conviction table and flat per-year table from HF.
    Returns (conviction_df, flat_df) — either may be None if not yet available.
    """
    try:
        from huggingface_hub import hf_hub_download
        from config import HF_RESULTS_DATASET

        token = token or os.environ.get("HF_TOKEN", "")

        def _load(fname):
            local = hf_hub_download(repo_id=HF_RESULTS_DATASET,
                                    filename=fname,
                                    repo_type="dataset",
                                    token=token or None)
            return pd.read_parquet(local)

        conviction = _load(CONSENSUS_FILE)
        flat       = _load(f"{CONSENSUS_DIR}/flat_latest.parquet")
        return conviction, flat
    except Exception:
        return None, None
