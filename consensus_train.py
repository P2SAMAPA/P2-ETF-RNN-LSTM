#!/usr/bin/env python3
# consensus_train.py
# Headless consensus sweep — called by GitHub Actions (or manually).
#
# Runs the ARMA-RNN-LSTM pipeline for every training-start year 2008–2024
# in parallel, scores conviction, and pushes results to HuggingFace.
#
# Usage:
#   python consensus_train.py                   # all years
#   python consensus_train.py --years 2015 2020 # specific years (testing)
#   python consensus_train.py --workers 6       # override thread count

import os
import sys
import logging
import warnings
import argparse
import io
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/consensus.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

import torch
from config import (
    TARGET_ETFS, TRAIN_SPLIT, LOOKBACK, SEED,
    HF_RESULTS_DATASET,
)
from data_loader import (
    load_all_data, build_feature_matrix, make_sequences,
    train_test_split_sequences, Normaliser,
)
from hurst import hurst_exponent, classify_memory
from trainer import train_pipeline
from huggingface_hub import HfApi, hf_hub_download

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_YEARS   = list(range(2008, 2025))   # 2008 → 2024 inclusive
CONSENSUS_DIR   = "consensus"
KEEP_RUNS       = 2                          # stamped files to retain per type
CONVICTION_W    = {"votes": 0.40, "ret": 0.35, "hurst": 0.25}

ETF_LABELS = {
    "TLT": "iShares 20yr Treasury",
    "LQD": "iShares Inv Grade Corp",
    "HYG": "iShares High Yield Bond",
    "VNQ": "Vanguard Real Estate",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
}


# ══════════════════════════════════════════════════════════════════════════════
# Per-year inference
# ══════════════════════════════════════════════════════════════════════════════

def _infer_one_year(year: int, data: dict, device: torch.device) -> dict | None:
    """
    Run full pipeline for a single training-start year.
    Returns {etf: result_dict} or None on failure.
    """
    try:
        cutoff = pd.Timestamp(f"{year}-01-01")
        price_f = data["price"][data["price"].index >= cutoff]
        ret_f   = data["ret"][data["ret"].index >= cutoff]

        if len(price_f) < 200:
            logger.warning(f"[{year}] Insufficient data ({len(price_f)} rows) — skipping")
            return None

        # Slice all data keys to cutoff
        data_y = {k: v[v.index >= cutoff] if isinstance(v, pd.DataFrame) else v
                  for k, v in data.items()}

        results = {}
        for etf in TARGET_ETFS:
            if etf not in price_f.columns:
                continue
            try:
                ret_s = (ret_f[etf].dropna().values
                         if etf in ret_f.columns
                         else np.diff(np.log(price_f[etf].dropna().values)))

                H   = hurst_exponent(ret_s)
                mem = classify_memory(H)

                features    = build_feature_matrix(data_y, etf)
                X, y, dates = make_sequences(features, lookback=LOOKBACK)

                if len(X) < 50:
                    logger.warning(f"[{year}] {etf}: only {len(X)} sequences — skipping")
                    continue

                X_tr, y_tr, _, X_te, y_te, _ = train_test_split_sequences(
                    X, y, dates, TRAIN_SPLIT)

                norm      = Normaliser()
                X_tr_n    = norm.fit_transform(X_tr)
                X_te_n    = norm.transform(X_te)

                res = train_pipeline(
                    X_train=X_tr_n, y_train=y_tr,
                    X_test=X_te_n,  y_test=y_te,
                    etf=f"{etf}_{year}",
                    use_hybrid=mem["use_hybrid"],
                    device=device,
                )

                pred_logret   = float(res["test_preds"][-1]) if len(res["test_preds"]) else 0.0
                current_price = float(price_f[etf].iloc[-1])
                dir_acc       = res["metrics"].get(
                    "hybrid_dir_acc", res["metrics"].get("rnn_dir_acc", 50.0))

                results[etf] = {
                    "year":            year,
                    "etf":             etf,
                    "H":               round(H, 4),
                    "model":           mem["model"],
                    "pred_ret_pct":    round(pred_logret * 100, 4),
                    "predicted_price": round(current_price * np.exp(pred_logret), 4),
                    "current_price":   round(current_price, 4),
                    "dir_acc":         round(dir_acc, 2),
                }
                logger.info(f"  [{year}] {etf}: {pred_logret*100:+.3f}% H={H:.3f} {mem['model']}")

            except Exception as e:
                logger.warning(f"  [{year}] {etf} failed: {e}")
                continue

        return results if results else None

    except Exception as e:
        logger.error(f"[{year}] Outer failure: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Conviction scorer
# ══════════════════════════════════════════════════════════════════════════════

def _compute_conviction(all_rows: list[dict], years_run: int) -> pd.DataFrame:
    """
    Score ETFs using weighted conviction formula.
    Score = 0.40 × vote_share + 0.35 × norm_avg_return + 0.25 × norm_avg_H
    """
    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame()

    # Top ETF per year (by predicted return)
    top_per_year = (df.sort_values("pred_ret_pct", ascending=False)
                      .groupby("year").first().reset_index()[["year", "etf"]])
    vote_counts  = top_per_year["etf"].value_counts()
    total_years  = len(top_per_year["year"].unique())

    agg = (df.groupby("etf")
             .agg(avg_pred_ret=("pred_ret_pct", "mean"),
                  avg_H=("H",            "mean"),
                  avg_dir_acc=("dir_acc", "mean"),
                  year_count=("year",     "count"))
             .reset_index())

    agg["votes"]      = agg["etf"].map(vote_counts).fillna(0).astype(int)
    agg["vote_share"] = agg["votes"] / total_years

    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 1e-9 else pd.Series([0.5] * len(s), index=s.index)

    agg["norm_ret"] = minmax(agg["avg_pred_ret"])
    agg["norm_H"]   = minmax(agg["avg_H"])

    agg["conviction"] = (CONVICTION_W["votes"] * agg["vote_share"]
                       + CONVICTION_W["ret"]   * agg["norm_ret"]
                       + CONVICTION_W["hurst"] * agg["norm_H"])

    agg["rank"]      = agg["conviction"].rank(ascending=False, method="first").astype(int)
    agg["label"]     = agg["etf"].map(ETF_LABELS).fillna("")
    agg["years_run"] = years_run
    agg["run_ts"]    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return agg.sort_values("rank").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace persistence
# ══════════════════════════════════════════════════════════════════════════════

def _upload_parquet(df: pd.DataFrame, path: str, token: str,
                    run_ts: str, api: HfApi):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=path,
        repo_id=HF_RESULTS_DATASET,
        repo_type="dataset",
        token=token,
        commit_message=f"[auto] Consensus — {run_ts}",
    )
    logger.info(f"Uploaded → {path}")


def _save_to_hf(conviction_df: pd.DataFrame, flat_df: pd.DataFrame,
                run_ts: str, token: str):
    """
    Push conviction + flat tables to HF.
    Keeps only KEEP_RUNS most-recent stamped files; deletes the rest.
    """
    api = HfApi()
    stamp = run_ts[:16].replace(":", "").replace("-", "").replace("T", "_")  # 20260310_2000

    # Stamped archive files
    conv_stamped = f"{CONSENSUS_DIR}/consensus_{stamp}.parquet"
    flat_stamped = f"{CONSENSUS_DIR}/flat_{stamp}.parquet"

    # Always-latest pointer files (overwritten every run)
    conv_latest  = f"{CONSENSUS_DIR}/consensus_latest.parquet"
    flat_latest  = f"{CONSENSUS_DIR}/flat_latest.parquet"

    _upload_parquet(conviction_df, conv_stamped, token, run_ts, api)
    _upload_parquet(conviction_df, conv_latest,  token, run_ts, api)
    _upload_parquet(flat_df,       flat_stamped, token, run_ts, api)
    _upload_parquet(flat_df,       flat_latest,  token, run_ts, api)

    # ── Clean up old stamped files ─────────────────────────────────────────
    try:
        from huggingface_hub import list_repo_files
        all_files = list(list_repo_files(HF_RESULTS_DATASET,
                                         repo_type="dataset", token=token))

        def _clean(prefix, keep_this):
            candidates = sorted([
                f for f in all_files
                if f.startswith(f"{CONSENSUS_DIR}/{prefix}_2")  # date-stamped only
            ])
            # Remove everything except the KEEP_RUNS most recent
            to_delete = candidates[:-KEEP_RUNS] if len(candidates) > KEEP_RUNS else []
            for old in to_delete:
                if old == keep_this:
                    continue
                try:
                    api.delete_file(path_in_repo=old,
                                    repo_id=HF_RESULTS_DATASET,
                                    repo_type="dataset", token=token,
                                    commit_message=f"[auto] Cleanup {old}")
                    logger.info(f"Deleted old file: {old}")
                except Exception as ex:
                    logger.warning(f"Could not delete {old}: {ex}")

        _clean("consensus", conv_stamped)
        _clean("flat",      flat_stamped)

    except Exception as e:
        logger.warning(f"Cleanup step failed (non-critical): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P2-ETF Consensus Sweep")
    parser.add_argument("--years",   nargs="+", type=int, default=DEFAULT_YEARS,
                        help="Training-start years to sweep (default: 2008-2024)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads (default: 4)")
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("=" * 70)
    logger.info("P2-ETF Consensus Sweep")
    logger.info(f"Run: {run_ts}")
    logger.info(f"Years: {args.years[0]}–{args.years[-1]} ({len(args.years)} windows)")
    logger.info(f"Workers: {args.workers}")
    logger.info("=" * 70)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        logger.error("HF_TOKEN not set — cannot load data or save results.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load full data once (shared across all year-threads) ──────────────────
    logger.info("\n── Loading data from HuggingFace ──")
    data = load_all_data(token)
    logger.info(f"Price data: {len(data['price'])} rows "
                f"({data['price'].index[0].date()} → {data['price'].index[-1].date()})")

    # ── Parallel sweep ────────────────────────────────────────────────────────
    logger.info(f"\n── Running {len(args.years)} year windows in parallel ──")
    all_flat   = []
    year_tops  = {}
    done = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_infer_one_year, yr, data, device): yr
                   for yr in args.years}

        for fut in as_completed(futures):
            yr = futures[fut]
            done += 1
            try:
                res = fut.result()
                if res:
                    rows = list(res.values())
                    all_flat.extend(rows)
                    top = max(rows, key=lambda r: r["pred_ret_pct"])
                    year_tops[yr] = top["etf"]
                    logger.info(f"✅ [{done:2d}/{len(args.years)}] {yr} → "
                                f"top: {top['etf']} ({top['pred_ret_pct']:+.3f}%)")
                else:
                    failed += 1
                    logger.warning(f"⚠️ [{done:2d}/{len(args.years)}] {yr} → no results")
            except Exception as e:
                failed += 1
                logger.error(f"❌ [{done:2d}/{len(args.years)}] {yr} → {e}")

    years_run = done - failed
    logger.info(f"\n── Sweep complete: {years_run} succeeded, {failed} failed ──")

    if not all_flat:
        logger.error("No results produced — aborting.")
        sys.exit(1)

    # ── Conviction scoring ────────────────────────────────────────────────────
    logger.info("\n── Computing conviction scores ──")
    conviction_df = _compute_conviction(all_flat, years_run)
    flat_df       = pd.DataFrame(all_flat)
    flat_df["run_ts"]    = run_ts
    flat_df["years_run"] = years_run

    # Log final ranking
    logger.info("\n📊 CONSENSUS RANKING:")
    logger.info(f"{'Rank':<5} {'ETF':<5} {'Conv':>7} {'Votes':>7} "
                f"{'Avg Ret%':>10} {'Avg H':>7} {'Dir Acc%':>9}")
    logger.info("-" * 55)
    for _, row in conviction_df.iterrows():
        logger.info(f"#{int(row['rank']):<4} {row['etf']:<5} "
                    f"{row['conviction']:>7.3f} "
                    f"{int(row['votes']):>4}/{years_run:<3} "
                    f"{row['avg_pred_ret']:>+9.3f}% "
                    f"{row['avg_H']:>7.3f} "
                    f"{row['avg_dir_acc']:>8.1f}%")

    top_etf = conviction_df.iloc[0]["etf"]
    top_conv = conviction_df.iloc[0]["conviction"]
    logger.info(f"\n★ CONSENSUS TOP PICK: {top_etf} "
                f"(conviction={top_conv:.3f}, "
                f"votes={int(conviction_df.iloc[0]['votes'])}/{years_run})")

    # ── Save to HuggingFace ───────────────────────────────────────────────────
    logger.info("\n── Saving to HuggingFace ──")
    _save_to_hf(conviction_df, flat_df, run_ts, token)

    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Consensus sweep complete — top pick: {top_etf}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
