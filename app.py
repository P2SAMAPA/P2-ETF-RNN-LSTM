# app.py — P2-ETF-RNN-LSTM (corrected hero box date)
import os
import warnings
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF RNN-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NYSE Calendar helper ───────────────────────────────────────────────────────
nyse = mcal.get_calendar("NYSE")

def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """
    Return the next NYSE trading day after the given date.
    """
    schedule = nyse.schedule(start_date=date, end_date=date + pd.Timedelta(days=10))
    trading_days = schedule.index
    next_days = trading_days[trading_days > date]
    if len(next_days) > 0:
        return next_days[0]
    # fallback (should not happen for valid dates)
    return date + pd.Timedelta(days=1)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_RESULTS   = "P2SAMAPA/p2-etf-rnn-lstm-results"
HF_SOURCE    = "P2SAMAPA/p2-etf-deepwave-dl"
TARGET_ETFS  = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
TRAIN_SPLIT  = 0.80
AUDIT_ROWS   = 15
HURST_THRESH = 0.52

ETF_LABELS = {
    "TLT": "iShares 20yr Treasury",
    "LQD": "iShares Inv Grade Corp",
    "HYG": "iShares High Yield Bond",
    "VNQ": "Vanguard Real Estate",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
}
ETF_COLORS = {
    "TLT": "#0ea5e9", "LQD": "#ec4899", "HYG": "#10b981",
    "VNQ": "#8b5cf6", "GLD": "#f59e0b", "SLV": "#94a3b8",
    "SPY": "#6366f1", "AGG": "#84cc16",
}

# Conviction weights (v2) — dir_acc replaces raw predicted return
CONVICTION_W = {"votes": 0.35, "dir_acc": 0.40, "hurst": 0.25}

# ── Shared Plotly layout helper ────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(color="#1e293b"),
)

# ── Helper: last trading day (for audit trail) ─────────────────────────────────
def last_trading_day_on_or_before(ref: pd.Timestamp) -> pd.Timestamp:
    """Return ref if it's a weekday, otherwise roll back to Friday."""
    d = ref
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=36)
    st.title("P2-ETF RNN-LSTM")
    st.caption("ARMA-RNN-LSTM Hybrid Neural Engine")
    st.caption(f"🕐 {datetime.now().strftime('%a %b %d, %H:%M')}")
    st.divider()

    st.subheader("⚙️ Configuration")
    start_year = st.slider("Training Data From (Year)", 2008, 2025, 2010,
                            help="Use data from this year onwards for training")
    st.divider()

    st.subheader("📊 Display")
    benchmark = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    st.divider()

    force_refresh = st.button("🔄 Force Data Refresh", use_container_width=True)
    run_inference = st.button("🚀 Run Live Inference", use_container_width=True,
                               type="primary")

    st.divider()
    st.subheader("🔁 Consensus Sweep")
    st.caption("Reads latest conviction results from HuggingFace (trained nightly via GitHub Actions).")
    refresh_consensus = st.button("🔄 Refresh Consensus Data",
                                   use_container_width=True,
                                   help="Re-fetch the latest consensus results from HuggingFace.")

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_results_parquet(filename: str) -> pd.DataFrame | None:
    token = os.environ.get("HF_TOKEN", "")
    try:
        local = hf_hub_download(repo_id=HF_RESULTS, filename=filename,
                                repo_type="dataset", token=token or None)
        df = pd.read_parquet(local)
        for col in ["date", "run_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_source_parquet(filename: str) -> pd.DataFrame | None:
    token = os.environ.get("HF_TOKEN", "")
    try:
        local = hf_hub_download(repo_id=HF_SOURCE, filename=filename,
                                repo_type="dataset", token=token or None)
        df = pd.read_parquet(local)
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_consensus_from_hf() -> tuple:
    try:
        token = os.environ.get("HF_TOKEN", "")
        def _load(fname):
            local = hf_hub_download(
                repo_id=HF_RESULTS, filename=fname,
                repo_type="dataset", token=token or None)
            return pd.read_parquet(local)
        conviction = _load("consensus/consensus_latest.parquet")
        flat       = _load("consensus/flat_latest.parquet")
        return conviction, flat
    except Exception:
        return None, None


def clear_cache():
    load_results_parquet.clear()
    load_source_parquet.clear()
    load_consensus_from_hf.clear()


if force_refresh:
    clear_cache()
    st.rerun()

if refresh_consensus:
    load_consensus_from_hf.clear()
    st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading latest results from HuggingFace…"):
    predictions = load_results_parquet("predictions.parquet")
    rankings    = load_results_parquet("rankings.parquet")
    metrics_df  = load_results_parquet("metrics.parquet")
    audit_df    = load_results_parquet("audit_trail.parquet")
    price_df    = load_source_parquet("data/etf_price.parquet")
    ret_df      = load_source_parquet("data/etf_ret.parquet")
    bench_ret   = load_source_parquet("data/bench_ret.parquet")
    bench_price = load_source_parquet("data/bench_price.parquet")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📈 P2-ETF RNN-LSTM Neural Forecasting Engine")
st.caption("ARMA-RNN-LSTM Hybrid Model · Xiao (2025) PLoS ONE · ETFs: " + " · ".join(TARGET_ETFS))

# ── Status badges ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
if price_df is not None:
    c1.success(f"✅ Dataset: {len(price_df):,} rows × {len(price_df.columns)} cols "
               f"({price_df.index[0].date()} → {price_df.index[-1].date()})")
else:
    c1.error("❌ Source data unavailable")

if predictions is not None:
    c2.success(f"✅ Predictions loaded ({len(predictions):,} rows)")
else:
    c2.warning("⚠️ No predictions yet — run training pipeline")

if rankings is not None:
    c3.success(f"✅ Rankings available — latest: {rankings['date'].max().strftime('%Y-%m-%d')}")
else:
    c3.warning("⚠️ No rankings yet")

if metrics_df is not None:
    c4.success(f"✅ Metrics loaded ({len(metrics_df):,} rows)")
else:
    c4.info("ℹ️ No metrics yet")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# LIVE INFERENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def _run_live_inference(price_df, ret_df, start_year):
    if price_df is None:
        st.error("Cannot run inference — price data not loaded.")
        return
    from data_loader import build_feature_matrix, make_sequences, \
        train_test_split_sequences, Normaliser
    from hurst import hurst_exponent, classify_memory
    from trainer import train_pipeline
    import torch

    cutoff  = pd.Timestamp(f"{start_year}-01-01")
    price_f = price_df[price_df.index >= cutoff]
    ret_f   = ret_df[ret_df.index >= cutoff] if ret_df is not None else None

    progress = st.progress(0, text="Starting inference…")
    results  = []
    device   = torch.device("cpu")

    for i, etf in enumerate(TARGET_ETFS):
        progress.progress(i / len(TARGET_ETFS), text=f"Processing {etf}…")
        if etf not in price_f.columns:
            continue
        data_full = {
            "price": price_f,
            "ret":   ret_f if ret_f is not None else pd.DataFrame(),
            "vol":   pd.DataFrame(index=price_f.index),
            "bench_price": pd.DataFrame(index=price_f.index),
            "bench_ret":   pd.DataFrame(index=price_f.index),
            "bench_vol":   pd.DataFrame(index=price_f.index),
        }
        ret_s = (ret_f[etf].dropna().values
                 if ret_f is not None and etf in ret_f.columns
                 else np.diff(np.log(price_f[etf].dropna().values)))
        H   = hurst_exponent(ret_s)
        mem = classify_memory(H)
        features = build_feature_matrix(data_full, etf)
        X, y, dates = make_sequences(features)
        if len(X) < 50:
            continue
        X_tr, y_tr, _, X_te, y_te, _ = train_test_split_sequences(X, y, dates, TRAIN_SPLIT)
        norm = Normaliser()
        res  = train_pipeline(norm.fit_transform(X_tr), y_tr,
                              norm.transform(X_te),     y_te,
                              etf, mem["use_hybrid"], device)
        current_price = float(price_f[etf].iloc[-1])
        pred_logret   = float(res["test_preds"][-1]) if len(res["test_preds"]) else 0.0
        results.append({
            "etf": etf, "H": H, "model": mem["model"],
            "pred_ret_pct":    pred_logret * 100,
            "predicted_price": current_price * np.exp(pred_logret),
            "current_price":   current_price,
            "dir_acc": res["metrics"].get("hybrid_dir_acc",
                       res["metrics"].get("rnn_dir_acc", 0)),
        })

    progress.progress(1.0, text="Done!")
    if results:
        results.sort(key=lambda x: x["pred_ret_pct"], reverse=True)
        st.session_state["live_results"] = results
        st.session_state["live_ran"]     = True
        st.success("✅ Live inference complete — signals updated below ↓")


if run_inference:
    _run_live_inference(price_df, ret_df, start_year)

# ── Determine signal source ────────────────────────────────────────────────────
live_ran     = st.session_state.get("live_ran", False)
live_results = st.session_state.get("live_results", [])

if live_ran and live_results:
    signal_source   = "🚀 Live Inference (in-browser)"
    # Live inference uses today as the base date → signal for next trading day
    _now = pd.Timestamp(datetime.now().date())
    signal_date = next_trading_day(_now)
    signal_date_str = signal_date.strftime("%A %b %d, %Y")
    latest_rankings = pd.DataFrame([
        {
            "rank":                 i + 1,
            "etf":                  r["etf"],
            "predicted_return_pct": r["pred_ret_pct"],
            "predicted_price":      r["predicted_price"],
            "current_price":        r["current_price"],
            "model_used":           r["model"],
            "hurst_H":              r["H"],
            "direction_accuracy":   r["dir_acc"],
            "date":                 pd.Timestamp(datetime.now().date()),
        }
        for i, r in enumerate(live_results)
    ])
elif rankings is not None and len(rankings) > 0:
    signal_source   = "📡 Last Scheduled Run (GitHub Actions)"
    # The pipeline already stores the target date in the rankings.
    # Use that date directly as the signal date.
    latest_rankings = rankings[rankings["date"] == rankings["date"].max()].sort_values("rank")
    if len(latest_rankings) > 0:
        signal_date = pd.Timestamp(latest_rankings.iloc[0]["date"])
        signal_date_str = signal_date.strftime("%A %b %d, %Y")
    else:
        signal_date_str = ""
else:
    latest_rankings = pd.DataFrame()
    signal_source   = ""
    signal_date_str = ""

# ── Top Signal Banner ──────────────────────────────────────────────────────────
if len(latest_rankings) > 0:
    top = latest_rankings.iloc[0]
    ret = top["predicted_return_pct"]
    ret_color = "#16a34a" if ret >= 0 else "#dc2626"
    bg_color  = "#f0fdf4" if ret >= 0 else "#fef2f2"
    bdr_color = "#bbf7d0" if ret >= 0 else "#fecaca"

    st.caption(f"Signal source: {signal_source}")

    st.markdown(f"""
    <div style="background:{bg_color}; border:2px solid {bdr_color};
                border-radius:12px; padding:22px 28px; margin-bottom:16px;">
        <div style="font-size:11px; letter-spacing:3px; color:#6b7280; margin-bottom:8px;">
            NEXT TRADING DAY SIGNAL — {signal_date_str}
            <span style="float:right; font-size:10px; color:#9ca3af;">
                MODEL: {top['model_used']}
            </span>
        </div>
        <div style="font-size:52px; font-weight:900; color:#111827; line-height:1;">
            {top['etf']}
        </div>
        <div style="font-size:13px; color:#6b7280; margin-top:4px;">
            {ETF_LABELS.get(top['etf'], '')}
        </div>
        <div style="margin-top:10px; font-size:14px; color:#374151;">
            Predicted Return:
            <strong style="color:{ret_color};">
                {'+' if ret >= 0 else ''}{ret:.3f}%
            </strong>
            &nbsp;·&nbsp; Dir Accuracy:
            <strong style="color:#d97706;">{top['direction_accuracy']:.1f}%</strong>
            &nbsp;·&nbsp; H =
            <strong style="color:#0284c7;">{top['hurst_H']:.3f}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 All ETF Predictions — Next Trading Day")
    pred_cols = st.columns(3)
    for i, row in latest_rankings.iterrows():
        cidx = int(row["rank"] - 1) % 3
        etf  = row["etf"]
        r    = row["predicted_return_pct"]
        ret_col = "#16a34a" if r > 0 else "#dc2626"
        bg      = "#f0fdf4" if r > 0 else "#fef2f2"
        bdr     = "#bbf7d0" if r > 0 else "#fecaca"
        etf_col = ETF_COLORS.get(etf, "#374151")
        with pred_cols[cidx]:
            st.markdown(f"""
            <div style="border:1px solid {bdr}; border-radius:10px;
                        padding:16px; margin-bottom:12px; background:{bg};">
                <div style="font-size:10px; color:#9ca3af; margin-bottom:4px;">
                    #{int(row['rank'])} · {ETF_LABELS.get(etf, etf)}
                </div>
                <div style="font-size:26px; font-weight:900; color:{etf_col};">{etf}</div>
                <div style="font-size:22px; font-weight:700; color:{ret_col}; margin:4px 0;">
                    {'+' if r > 0 else ''}{r:.3f}%
                </div>
                <div style="font-size:11px; color:#6b7280;">
                    H={row['hurst_H']:.3f} · {row['model_used']}
                    · Dir Acc {row['direction_accuracy']:.1f}%
                </div>
                <div style="font-size:11px; color:#9ca3af; margin-top:4px;">
                    ${row['current_price']:.2f} → ${row['predicted_price']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Tabs (unchanged from here) ─────────────────────────────────────────────────
# ... [rest of the tabs, same as your original file] ...
# To keep the message within limits, I'm truncating here.
# Please keep the remainder of your original app.py unchanged.
# Only the hero box date logic was modified.
