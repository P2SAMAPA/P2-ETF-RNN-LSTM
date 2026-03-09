# 📈 P2-ETF-RNN-LSTM

**ARMA-RNN-LSTM Hybrid Neural Forecasting Engine for ETF Next-Day Signal Generation**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-RNN-LSTM/actions/workflows/train.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-RNN-LSTM/actions)
[![Hugging Face](https://img.shields.io/badge/🤗%20HF-p2--etf--rnn--lstm--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-rnn-lstm-results)

---

## 🔬 The Paper: Xiao (2025)

> **Xiao H (2025)** — *Enhanced separation of long-term memory from short-term memory on top of LSTM: Neural network-based stock index forecasting.*
> PLoS ONE 20(6): e0322737. https://doi.org/10.1371/journal.pone.0322737

### The Core Problem

LSTM networks are designed to separate long-term from short-term memory via gating. However, Xiao (2025) demonstrates that **LSTMs often misclassify short-term memory as long-term**, reducing forecast accuracy. This is evidenced by plain RNNs outperforming LSTMs on series that contain only short-term memory — the LSTM wastes capacity searching for long-term patterns that do not exist.

### The Solution: External Memory Separation (ARMA-RNN-LSTM)

Pre-separate the two memory types **before** the data enters the LSTM:

| Stage | Model | Role | Paper Equation |
|-------|-------|------|----------------|
| **1** | SimpleRNN | Captures **short-term memory** (RNNs structurally cannot do long-term due to vanishing gradients) | Eq. 24–27 |
| **2** | ResidualLSTM | Trained on RNN residuals → extracts **long-term memory** from what RNN could not explain | Eq. 28–29 |
| **3** | HybridLSTM | Integrates `[features ∣ ŷ_RNN ∣ ŷ_LSTM]` → final refined forecast | Eq. 30–33 |

This mirrors the **ARMA structure**: AR component ≡ RNN (short-term), MA component ≡ Residual LSTM (error/long-term).

### Hurst Exponent Decision Rule

Before training, R/S analysis determines which model to use:

```
H > 0.52  →  Long-term memory detected  →  ARMA-RNN-LSTM hybrid
H ≈ 0.50  →  Near random walk           →  Standalone RNN (paper §4.3)
H < 0.45  →  Anti-persistent            →  Standalone RNN
```

### Key Paper Results

| Index | Hurst H | Model Used | MAE |
|-------|---------|------------|-----|
| SZSE (Shenzhen) | 0.63 | ARMA-RNN-LSTM | 24.79 |
| HSI (Hang Seng) | 0.57 | ARMA-RNN-LSTM | 41.89 |
| SSE (Shanghai) | 0.52 | RNN only | RNN wins |

---

## 🏗️ What This Engine Does

A **production daily pipeline** applying the paper to 6 ETFs, running automatically after US market close every weekday.

### ETFs

| Ticker | Name | Asset Class |
|--------|------|-------------|
| TLT | iShares 20yr Treasury | Long-duration Treasuries |
| LQD | iShares IG Corp Bond | Investment Grade Bonds |
| HYG | iShares High Yield Bond | High Yield Bonds |
| VNQ | Vanguard Real Estate | US REITs |
| GLD | SPDR Gold Shares | Gold |
| SLV | iShares Silver Trust | Silver |

### Daily Pipeline Flow

```
HF Source: P2SAMAPA/p2-etf-deepwave-dl
  etf_price.parquet | etf_ret.parquet | etf_vol.parquet
  bench_price.parquet | bench_ret.parquet (SPY, AGG)
          │
          ▼
  1. Hurst Exponent (R/S Analysis) per ETF
          │
          ▼
  2. Feature Engineering
     log-returns + rolling vol + volume z-score + SPY/AGG context
          │
          ▼
  3. ARMA-RNN-LSTM 3-Stage Training (or RNN-only if H ≈ 0.5)
          │
          ▼
  4. Next-Day Predictions + Rankings (BUY / LONG / AVOID)
          │
          ▼
HF Output: P2SAMAPA/p2-etf-rnn-lstm-results
  predictions.parquet | rankings.parquet
  metrics.parquet | audit_trail.parquet | weights/
          │
          ▼
  Streamlit Dashboard (live inference + results display)
```

---

## 📁 Repository Structure

```
P2-ETF-RNN-LSTM/
├── .github/
│   └── workflows/
│       └── train.yml       # Daily cron: 21:00 UTC = 4:00 PM ET, Mon–Fri
│
├── train.py                # Main pipeline orchestrator (entry point)
├── models.py               # PyTorch: SimpleRNN, ResidualLSTM, HybridLSTM
├── trainer.py              # 3-stage sequential training logic + metrics
├── data_loader.py          # HF data loading, feature engineering, sequences
├── hurst.py                # Hurst exponent via R/S analysis (paper §2.1)
├── hf_io.py                # HuggingFace read/write for all output files
├── app.py                  # Streamlit dashboard (UI + live inference)
├── config.py               # All constants, hyperparameters, file paths
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup

### 1. GitHub Secret

```
Settings → Secrets and variables → Actions → New repository secret
Name:  HF_TOKEN
Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. HuggingFace Datasets

| Dataset | Role |
|---------|------|
| `P2SAMAPA/p2-etf-deepwave-dl` | **Source** — OHLC + return + volume data (2008–present) |
| `P2SAMAPA/p2-etf-rnn-lstm-results` | **Output** — predictions, rankings, metrics, weights |

Create `p2-etf-rnn-lstm-results` as a **public** dataset repo on HuggingFace.
It will be populated automatically on first pipeline run.

### 3. Streamlit Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io) → New app
2. Repo: `P2SAMAPA/P2-ETF-RNN-LSTM`, Branch: `main`, File: `app.py`
3. Add secret: `HF_TOKEN = hf_xxxx`

### 4. Run Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-RNN-LSTM.git
cd P2-ETF-RNN-LSTM
pip install -r requirements.txt
export HF_TOKEN=hf_xxxx

# Run training pipeline
python train.py

# Launch Streamlit app
streamlit run app.py
```

### 5. Manual Training Trigger

```
GitHub → Actions → Daily ETF Training Pipeline → Run workflow
Options:
  retrain_from_scratch: true/false
  lookback_years: all / 5 / 10
```

---

## 🔧 Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOOKBACK` | 10 | Sequence length (paper uses similar window) |
| `TRAIN_SPLIT` | 0.625 | Train/test split — matches paper (250/400) |
| `RNN_HIDDEN` | 64 | RNN hidden units |
| `LSTM_HIDDEN` | 128 | LSTM hidden units |
| `LSTM2_HIDDEN` | 128 | Final hybrid LSTM hidden units |
| `EPOCHS_RNN` | 100 | Stage 1 training epochs |
| `EPOCHS_LSTM1` | 100 | Stage 2 training epochs |
| `EPOCHS_LSTM2` | 150 | Stage 3 training epochs |
| `EARLY_STOP_PATIENCE` | 15 | Early stopping patience |
| `LONG_MEMORY_THRESH` | 0.52 | Hurst H threshold |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `GRAD_CLIP` | 1.0 | Gradient clipping norm |

---

## 📊 Output Dataset Schema

### `predictions.parquet`
| Column | Description |
|--------|-------------|
| `date` | Prediction target date (next trading day) |
| `etf` | ETF ticker |
| `current_price` | Last known closing price |
| `predicted_return_pct` | Predicted next-day return (%) |
| `predicted_price` | Predicted next-day price |
| `model_used` | "ARMA-RNN-LSTM" or "RNN" |
| `hurst_H` | Hurst exponent H |
| `memory_type` | "long" / "random" / "anti-persistent" |
| `direction_accuracy` | Test-set directional accuracy (%) |
| `mae` | Test-set Mean Absolute Error |
| `rmse` | Test-set Root Mean Squared Error |
| `run_timestamp` | UTC timestamp of training run |

### `rankings.parquet`
Daily ranking of all 6 ETFs by predicted return with BUY / LONG / AVOID signals.

### `metrics.parquet`
Per-ETF per-run training metrics for all model variants (hybrid vs RNN vs combined).

### `audit_trail.parquet`
Full signal history with `predicted_ret_pct` and `actual_ret_pct` backfilled the following day — enables live P&L tracking.

### `weights/`
PyTorch state dicts: `{etf}_rnn.pt`, `{etf}_residual_lstm.pt`, `{etf}_hybrid_lstm.pt`

---

## 📄 Streamlit App Sections

| Tab | Contents |
|-----|----------|
| **Next-Day Signal** | Top pick banner + all 6 ETF predictions ranked |
| **Forecast Charts** | Price history + benchmark overlay + prediction markers |
| **Model Performance** | Directional accuracy, MAE comparison across model variants |
| **Hurst Analysis** | H values per ETF, model selection rationale |
| **Audit Trail** | Last N days of signals + actual returns + win rate |
| **About** | Paper summary, pipeline explanation, data schema |

---

## ⚠️ Disclaimer

For **educational and research purposes only**. Implements the methodology from Xiao (2025). Does not constitute financial advice. Past model performance does not guarantee future results.

---

## 📜 Citation

```bibtex
@article{xiao2025lstm,
  title   = {Enhanced separation of long-term memory from short-term memory
             on top of LSTM: Neural network-based stock index forecasting},
  author  = {Xiao, Hongfei},
  journal = {PLoS ONE},
  volume  = {20},
  number  = {6},
  pages   = {e0322737},
  year    = {2025},
  doi     = {10.1371/journal.pone.0322737}
}
```
