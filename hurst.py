# hurst.py
# Hurst Exponent via R/S Analysis — directly from Xiao (2025) Section 2.1
#
# H = 0.5        → random walk (no memory)
# 0 < H < 0.5    → anti-persistent (mean-reverting)
# 0.5 < H < 1    → long-term memory (trending) → use ARMA-RNN-LSTM hybrid
#
# R/S method: Mandelbrot (1972), Hurst (1951)

import numpy as np
import logging
from config import HURST_MIN_WINDOW, HURST_MAX_RATIO, HURST_STEP, LONG_MEMORY_THRESH

logger = logging.getLogger(__name__)


def hurst_exponent(series: np.ndarray) -> float:
    """
    Compute the Hurst exponent via R/S analysis.

    Implements Equations (1)–(7) from Xiao (2025):
      1. Divide series into blocks of size n
      2. Compute mean per block: x̄_n = (1/n) Σ x_i
      3. Compute range R(n) = max(cumdev) - min(cumdev)
      4. Compute std S(n)
      5. Q_n = R(n)/S(n)
      6. Q_n = C * n^H  →  ln(Q_n) = ln(C) + H * ln(n)
      7. H ≈ slope of OLS regression of ln(Q_n) on ln(n)

    Parameters
    ----------
    series : np.ndarray
        1-D array of returns (or prices — returns preferred for stationarity)

    Returns
    -------
    float : Hurst exponent H in (0, 1)
    """
    series = np.asarray(series, dtype=np.float64)
    series = series[~np.isnan(series)]
    n_total = len(series)

    if n_total < 20:
        logger.warning("Series too short for reliable Hurst estimation, returning 0.5")
        return 0.5

    log_ns, log_rs = [], []

    size = HURST_MIN_WINDOW
    while size <= n_total * HURST_MAX_RATIO:
        n = int(size)
        n_blocks = n_total // n
        if n_blocks < 1:
            break

        rs_vals = []
        for b in range(n_blocks):
            chunk = series[b * n: (b + 1) * n]
            mean  = np.mean(chunk)
            devs  = chunk - mean
            cum   = np.cumsum(devs)
            R     = cum.max() - cum.min()
            S     = np.std(chunk, ddof=0) + 1e-10
            rs_vals.append(R / S)

        mean_rs = np.mean(rs_vals)
        if mean_rs > 0:
            log_ns.append(np.log(n))
            log_rs.append(np.log(mean_rs))

        size = max(size + 1, int(size * HURST_STEP))

    if len(log_ns) < 3:
        logger.warning("Not enough R/S points, returning 0.5")
        return 0.5

    # OLS regression: H = slope (Eq. 6-7 in paper)
    log_ns = np.array(log_ns)
    log_rs = np.array(log_rs)
    mean_x = log_ns.mean()
    mean_y = log_rs.mean()
    num    = np.sum((log_ns - mean_x) * (log_rs - mean_y))
    den    = np.sum((log_ns - mean_x) ** 2)
    H      = num / den if den != 0 else 0.5

    H = float(np.clip(H, 0.01, 0.99))
    return H


def classify_memory(H: float) -> dict:
    """
    Classify the memory type of a series based on its Hurst exponent.

    Returns a dict with:
      - memory_type: 'long' | 'anti-persistent' | 'random'
      - use_hybrid:  bool — True means use ARMA-RNN-LSTM hybrid
      - description: human-readable string
    """
    if H > LONG_MEMORY_THRESH:
        return {
            "memory_type": "long",
            "use_hybrid":  True,
            "description": f"H={H:.3f} — Long-term memory detected. "
                           f"ARMA-RNN-LSTM hybrid model recommended (paper §3).",
            "model":       "ARMA-RNN-LSTM Hybrid",
        }
    elif H < 0.45:
        return {
            "memory_type": "anti-persistent",
            "use_hybrid":  False,
            "description": f"H={H:.3f} — Anti-persistent (mean-reverting). "
                           f"Standalone RNN recommended.",
            "model":       "RNN",
        }
    else:
        return {
            "memory_type": "random",
            "use_hybrid":  False,
            "description": f"H={H:.3f} — Near random walk. "
                           f"Standalone RNN recommended (paper §4.3).",
            "model":       "RNN",
        }


def compute_hurst_all_etfs(returns: dict) -> dict:
    """
    Compute Hurst exponent for all ETFs.

    Parameters
    ----------
    returns : dict[str, np.ndarray]  — ETF ticker → return series

    Returns
    -------
    dict[str, dict]  — ETF ticker → {'H': float, 'memory_type': str, ...}
    """
    results = {}
    for etf, ret_series in returns.items():
        H    = hurst_exponent(np.asarray(ret_series))
        info = classify_memory(H)
        info["H"] = H
        results[etf] = info
        logger.info(f"  {etf}: H={H:.3f} → {info['memory_type']} memory → {info['model']}")
    return results
