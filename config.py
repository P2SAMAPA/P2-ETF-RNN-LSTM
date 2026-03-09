# config.py
# Central configuration for P2-ETF-RNN-LSTM project

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_SOURCE_DATASET   = "P2SAMAPA/p2-etf-deepwave-dl"
HF_RESULTS_DATASET  = "P2SAMAPA/p2-etf-rnn-lstm-results"

# ── ETFs ──────────────────────────────────────────────────────────────────────
TARGET_ETFS         = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARK_TICKERS   = ["SPY", "AGG"]

# ── Source parquet files (inside HF dataset /data/ folder) ───────────────────
ETF_PRICE_FILE      = "data/etf_price.parquet"
ETF_RET_FILE        = "data/etf_ret.parquet"
ETF_VOL_FILE        = "data/etf_vol.parquet"
BENCH_PRICE_FILE    = "data/bench_price.parquet"
BENCH_RET_FILE      = "data/bench_ret.parquet"
BENCH_VOL_FILE      = "data/bench_vol.parquet"

# ── Output parquet files (pushed to HF results dataset) ──────────────────────
OUT_PREDICTIONS     = "predictions.parquet"
OUT_RANKINGS        = "rankings.parquet"
OUT_METRICS         = "metrics.parquet"
OUT_AUDIT           = "audit_trail.parquet"
OUT_WEIGHTS_DIR     = "weights"          # folder inside results dataset

# ── Model architecture (from Xiao 2025) ──────────────────────────────────────
LOOKBACK            = 10        # sequence length fed into RNN/LSTM
TRAIN_SPLIT         = 0.625     # 62.5% train / 37.5% test (paper: 250/400)
RNN_HIDDEN          = 64
LSTM_HIDDEN         = 128
LSTM2_HIDDEN        = 128       # final hybrid LSTM hidden size
RNN_LAYERS          = 2
LSTM_LAYERS         = 2
DROPOUT             = 0.2
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-5
EPOCHS_RNN          = 100
EPOCHS_LSTM1        = 100       # LSTM on residuals
EPOCHS_LSTM2        = 150       # final hybrid LSTM
BATCH_SIZE          = 32
EARLY_STOP_PATIENCE = 15
GRAD_CLIP           = 1.0

# ── Hurst exponent ────────────────────────────────────────────────────────────
HURST_MIN_WINDOW    = 10
HURST_MAX_RATIO     = 0.5       # max block size = len(series) * ratio
HURST_STEP          = 1.5       # geometric step for R/S block sizes
LONG_MEMORY_THRESH  = 0.52      # H > 0.52 → use hybrid model; else plain RNN

# ── Features used as model input ─────────────────────────────────────────────
# Paper uses price series directly → we use log-returns + rolling vol
USE_LOG_RETURNS     = True
USE_ROLLING_VOL     = True      # 5-day rolling std of returns
USE_VOLUME          = True      # normalised volume from etf_vol
USE_BENCH           = True      # SPY + AGG returns as context features

# ── Training data window ──────────────────────────────────────────────────────
MIN_TRAIN_YEARS     = 2         # minimum years needed before first prediction
ROLLING_RETRAIN     = False     # if True, use expanding window; else full history

# ── Scheduler ────────────────────────────────────────────────────────────────
# GitHub Actions cron: '0 21 * * 1-5'  = 21:00 UTC = 4:00 PM ET weekdays

# ── Random seed ──────────────────────────────────────────────────────────────
SEED                = 42

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR             = "logs"
LOG_FILE            = "logs/training.log"
