# data_loader.py
# Loads ETF price, return, volume data from HuggingFace dataset
# Source: P2SAMAPA/p2-etf-deepwave-dl

import os
import io
import logging
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from config import (
    HF_SOURCE_DATASET, TARGET_ETFS, BENCHMARK_TICKERS,
    ETF_PRICE_FILE, ETF_RET_FILE, ETF_VOL_FILE,
    BENCH_PRICE_FILE, BENCH_RET_FILE, BENCH_VOL_FILE,
    USE_LOG_RETURNS, USE_ROLLING_VOL, USE_VOLUME, USE_BENCH,
    LOOKBACK, SEED
)

logger = logging.getLogger(__name__)


def _load_parquet(filename: str, token: str) -> pd.DataFrame:
    """Download a parquet file from HF dataset and return as DataFrame."""
    logger.info(f"Loading {filename} from {HF_SOURCE_DATASET}")
    local_path = hf_hub_download(
        repo_id=HF_SOURCE_DATASET,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(local_path)
    # Normalise date column
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "Date"})
        df = df.set_index("Date").sort_index()
    return df


def load_all_data(token: str) -> dict:
    """
    Load all source parquet files and return a dict of DataFrames.

    Returns
    -------
    {
        'price':       pd.DataFrame  [Date x ETF+VCIT],
        'ret':         pd.DataFrame  [Date x ETF+VCIT],
        'vol':         pd.DataFrame  [Date x ETF+VCIT],
        'bench_price': pd.DataFrame  [Date x SPY+AGG],
        'bench_ret':   pd.DataFrame  [Date x SPY+AGG],
        'bench_vol':   pd.DataFrame  [Date x SPY+AGG],
    }
    """
    data = {}
    data["price"]       = _load_parquet(ETF_PRICE_FILE,   token)
    data["ret"]         = _load_parquet(ETF_RET_FILE,     token)
    data["vol"]         = _load_parquet(ETF_VOL_FILE,     token)
    data["bench_price"] = _load_parquet(BENCH_PRICE_FILE, token)
    data["bench_ret"]   = _load_parquet(BENCH_RET_FILE,   token)
    data["bench_vol"]   = _load_parquet(BENCH_VOL_FILE,   token)

    # Keep only target ETFs (drop VCIT)
    for key in ["price", "ret", "vol"]:
        cols = [c for c in data[key].columns if c in TARGET_ETFS]
        data[key] = data[key][cols]

    # Keep only benchmark tickers
    for key in ["bench_price", "bench_ret", "bench_vol"]:
        cols = [c for c in data[key].columns if c in BENCHMARK_TICKERS]
        data[key] = data[key][cols]

    # Align all on common dates (inner join)
    common_idx = data["price"].index
    for key in data:
        common_idx = common_idx.intersection(data[key].index)
    for key in data:
        data[key] = data[key].loc[common_idx]

    logger.info(
        f"Data loaded: {len(common_idx)} trading days "
        f"({common_idx[0].date()} → {common_idx[-1].date()})"
    )
    return data


def build_feature_matrix(data: dict, etf: str) -> pd.DataFrame:
    """
    Build the feature matrix for a single ETF following the paper:

    Features (per Xiao 2025):
      - Log-return of the target ETF         (primary input)
      - 5-day rolling vol of target ETF      (short-term vol proxy)
      - Normalised volume of target ETF      (if available)
      - SPY log-return                        (market context)
      - AGG log-return                        (bond market context)

    Target:
      - Next-day log-return of target ETF
    """
    price = data["price"][etf].dropna()

    # Log returns
    log_ret = np.log(price / price.shift(1)).dropna()
    log_ret.name = f"{etf}_logret"

    features = pd.DataFrame(index=log_ret.index)
    features[f"{etf}_logret"] = log_ret

    # Rolling 5-day volatility
    if USE_ROLLING_VOL:
        roll_vol = log_ret.rolling(5).std()
        roll_vol.name = f"{etf}_rollvol"
        features[f"{etf}_rollvol"] = roll_vol

    # Volume (normalised z-score)
    if USE_VOLUME and etf in data["vol"].columns:
        vol_series = data["vol"][etf]
        vol_z = (vol_series - vol_series.rolling(60).mean()) / \
                (vol_series.rolling(60).std() + 1e-8)
        vol_z.name = f"{etf}_vol_z"
        features[f"{etf}_vol_z"] = vol_z.reindex(features.index)

    # Benchmark context features
    if USE_BENCH:
        for bench in BENCHMARK_TICKERS:
            if bench in data["bench_price"].columns:
                bp = data["bench_price"][bench].dropna()
                br = np.log(bp / bp.shift(1)).dropna()
                br.name = f"{bench}_logret"
                features[f"{bench}_logret"] = br.reindex(features.index)

    # Drop NaN rows from rolling calculations
    features = features.dropna()

    # Target: next-day log return
    features["target"] = features[f"{etf}_logret"].shift(-1)
    features = features.dropna()

    return features


def make_sequences(features: pd.DataFrame, lookback: int = LOOKBACK):
    """
    Convert feature DataFrame into (X, y) sliding-window sequences.

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples,)
    dates:   (n_samples,)  — date of the prediction target
    """
    feat_cols = [c for c in features.columns if c != "target"]
    X_raw = features[feat_cols].values
    y_raw = features["target"].values
    dates = features.index

    X, y, d = [], [], []
    for i in range(lookback, len(X_raw)):
        X.append(X_raw[i - lookback: i])
        y.append(y_raw[i])
        d.append(dates[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(d)


def train_test_split_sequences(X, y, dates, split: float = 0.625):
    """Split sequences into train/test preserving temporal order."""
    n = len(X)
    cut = int(n * split)
    return (
        X[:cut], y[:cut], dates[:cut],
        X[cut:], y[cut:], dates[cut:]
    )


class Normaliser:
    """Min-max normaliser fitted on training set only (no data leakage)."""

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X: np.ndarray):
        # X shape: (n_samples, lookback, n_features) or (n_samples, n_features)
        flat = X.reshape(-1, X.shape[-1])
        self.min_   = flat.min(axis=0)
        self.max_   = flat.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0,
                               self.max_ - self.min_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform_scalar(self, val: float, feature_idx: int = 0) -> float:
        return val * self.range_[feature_idx] + self.min_[feature_idx]
