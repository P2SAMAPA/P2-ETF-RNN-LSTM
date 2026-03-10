# app.py  — P2-ETF-RNN-LSTM  (updated UI + Consensus Sweep tab)

import os
import warnings
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF RNN-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# White background for all charts
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(color="#1e293b"),
)


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
    c4.success(f"✅ Metrics loaded ({len(metrics_df):,} runs)")
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
    signal_date_str = datetime.now().strftime("%A %b %d, %Y")
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
    latest_rankings = rankings[rankings["date"] == rankings["date"].max()].sort_values("rank")
    if len(latest_rankings) > 0:
        run_date = latest_rankings.iloc[0]["date"]
        from datetime import timedelta
        target = run_date + timedelta(days=1)
        while target.weekday() >= 5:
            target += timedelta(days=1)
        signal_date_str = target.strftime("%A %b %d, %Y")
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

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab_consensus, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 OOS Forecast Chart",
    "🔁 Consensus Sweep",
    "📊 Model Performance",
    "🌊 Hurst Analysis",
    "📋 Audit Trail",
    "ℹ️ About the Model",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB — CONSENSUS SWEEP
# ══════════════════════════════════════════════════════════════════════════════
with tab_consensus:
    st.subheader("🔁 Consensus Sweep — All Training Windows (2008–2024)")
    st.caption(
        "Each year from 2008 to 2024 is used as a training-start cutoff. "
        "All 17 windows run in parallel via GitHub Actions and their signals are "
        "combined using a weighted conviction score: "
        f"{int(CONVICTION_W['votes']*100)}% vote share · "
        f"{int(CONVICTION_W['dir_acc']*100)}% OOS directional accuracy · "
        f"{int(CONVICTION_W['hurst']*100)}% avg Hurst H."
    )

    conviction_df, flat_df = load_consensus_from_hf()

    if conviction_df is not None and len(conviction_df) > 0:
        run_ts    = conviction_df["run_ts"].iloc[0] if "run_ts" in conviction_df.columns else "unknown"
        years_run = int(conviction_df["years_run"].iloc[0]) if "years_run" in conviction_df.columns else "?"
        year_tops = {}
        if flat_df is not None and "year" in flat_df.columns and "pred_ret_pct" in flat_df.columns:
            year_tops = (
                flat_df.sort_values("pred_ret_pct", ascending=False)
                       .groupby("year").first()["etf"]
                       .to_dict()
            )
        data_source = f"📡 GitHub Actions run — {run_ts[:10] if run_ts != 'unknown' else '—'} UTC"
    else:
        conviction_df = None
        flat_df       = None
        run_ts        = None
        years_run     = 0
        year_tops     = {}
        data_source   = None

    # ── No data state ─────────────────────────────────────────────────────────
    if conviction_df is None or len(conviction_df) == 0:
        st.info(
            "No consensus results yet. The GitHub Actions workflow runs automatically "
            "at **8 PM EST weekdays**. You can also trigger it manually from the "
            "**Actions tab** in your GitHub repo → *Consensus Sweep* → *Run workflow*."
        )
        st.markdown("""
        **How it works:**
        1. Trains the ARMA-RNN-LSTM pipeline for each start year (2008 → 2024) in parallel threads
        2. Each window produces a top-ranked ETF for the next trading day
        3. Signals are combined using a weighted conviction formula:
           - **35%** — vote share (how many windows agree on this ETF as #1)
           - **40%** — avg OOS directional accuracy (predicted vs actual direction sync)
           - **25%** — avg Hurst H (trend-persistence signal strength)
        4. Results are saved to HuggingFace with a date stamp and cleaned up automatically
        """)
    else:
        st.caption(f"Source: {data_source} · {years_run} windows ran · Date: {run_ts[:10] if run_ts else '—'}")

        top_row    = conviction_df.iloc[0]
        top_etf    = top_row["etf"]
        conv       = float(top_row["conviction"])
        avg_dir    = float(top_row["avg_dir_acc"])
        avg_ret    = float(top_row["avg_pred_ret"])   # display only
        votes      = int(top_row["votes"])
        vote_share = float(top_row["vote_share"])
        avg_H      = float(top_row["avg_H"])

        ret_color = "#16a34a" if avg_ret >= 0 else "#dc2626"
        bg_color  = "#f0fdf4"
        bdr_color = "#bbf7d0"
        etf_color = ETF_COLORS.get(top_etf, "#111827")

        # ── Consensus hero banner ──────────────────────────────────────────
        signal_date_str = datetime.now().strftime("%A %b %d, %Y")
        st.markdown(f"""
        <div style="background:{bg_color}; border:2px solid {bdr_color};
                    border-radius:14px; padding:26px 32px; margin-bottom:20px;">
            <div style="font-size:11px; letter-spacing:3px; color:#6b7280; margin-bottom:8px;">
                CONSENSUS SIGNAL — {signal_date_str}
                <span style="float:right; font-size:10px; color:#9ca3af;">
                    CONVICTION SWEEP · 2008–2024
                </span>
            </div>
            <div style="font-size:58px; font-weight:900; color:{etf_color}; line-height:1;">
                {top_etf}
            </div>
            <div style="font-size:14px; color:#6b7280; margin-top:4px;">
                {ETF_LABELS.get(top_etf, '')}
            </div>
            <div style="margin-top:14px; display:flex; gap:32px; flex-wrap:wrap;">
                <div>
                    <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Conviction Score</div>
                    <div style="font-size:28px; font-weight:800; color:#7c3aed;">{conv:.3f}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Vote Share</div>
                    <div style="font-size:28px; font-weight:800; color:#0284c7;">{votes}/{years_run} ({vote_share:.0%})</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Avg OOS Dir Accuracy</div>
                    <div style="font-size:28px; font-weight:800; color:#d97706;">{avg_dir:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Avg Hurst H</div>
                    <div style="font-size:28px; font-weight:800; color:#0ea5e9;">{avg_H:.3f}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#9ca3af; text-transform:uppercase; letter-spacing:1px;">Avg Pred Return</div>
                    <div style="font-size:28px; font-weight:800; color:{ret_color};">{'+' if avg_ret >= 0 else ''}{avg_ret:.3f}%</div>
                    <div style="font-size:10px; color:#9ca3af;">display only · not in score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── All ETF conviction cards ───────────────────────────────────────
        st.subheader("📊 Full ETF Conviction Rankings")
        conv_cols = st.columns(3)
        for _, crow in conviction_df.iterrows():
            cidx    = (int(crow["rank"]) - 1) % 3
            cetf    = crow["etf"]
            cconv   = float(crow["conviction"])
            cvotes  = int(crow["votes"])
            cshare  = float(crow["vote_share"])
            cH      = float(crow["avg_H"])
            cdiracc = float(crow.get("avg_dir_acc", 0))
            cret    = float(crow.get("avg_pred_ret", 0))

            ec  = ETF_COLORS.get(cetf, "#374151")
            bg  = "#faf5ff" if int(crow["rank"]) == 1 else "#f8fafc"
            bdr = "#c4b5fd" if int(crow["rank"]) == 1 else "#e2e8f0"

            with conv_cols[cidx]:
                st.markdown(f"""
                <div style="border:1px solid {bdr}; border-radius:10px;
                            padding:16px; margin-bottom:12px; background:{bg};">
                    <div style="font-size:10px; color:#9ca3af; margin-bottom:4px;">
                        #{int(crow['rank'])} · {ETF_LABELS.get(cetf, cetf)}
                    </div>
                    <div style="font-size:26px; font-weight:900; color:{ec};">{cetf}</div>
                    <div style="font-size:18px; font-weight:700; color:#7c3aed; margin:4px 0;">
                        ⚡ {cconv:.3f} conviction
                    </div>
                    <div style="font-size:13px; color:#d97706; font-weight:600;">
                        Dir Acc: {cdiracc:.1f}%
                    </div>
                    <div style="font-size:11px; color:#6b7280; margin-top:4px;">
                        Votes: {cvotes}/{years_run} ({cshare:.0%}) · H={cH:.3f}
                    </div>
                    <div style="font-size:11px; color:#9ca3af; margin-top:2px;">
                        Avg pred return: {'+' if cret >= 0 else ''}{cret:.3f}% <em>(display only)</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Year-by-year vote heatmap (WHITE background) ───────────────────
        if year_tops:
            st.subheader("🗳️ Year-by-Year Top Picks")
            st.caption("Which ETF ranked #1 for each training-start year window")

            years_sorted = sorted(year_tops.keys())
            heat_etfs    = TARGET_ETFS

            z_colored = []
            for i, etf in enumerate(heat_etfs):
                row = [(i + 1) if year_tops.get(yr) == etf else 0 for yr in years_sorted]
                z_colored.append(row)

            fig_heat = go.Figure(go.Heatmap(
                z=z_colored,
                x=[str(yr) for yr in years_sorted],
                y=heat_etfs,
                colorscale=[
                    [0.0,  "#f1f5f9"],
                    [0.17, "#0ea5e9"],
                    [0.33, "#ec4899"],
                    [0.50, "#10b981"],
                    [0.67, "#8b5cf6"],
                    [0.83, "#f59e0b"],
                    [1.0,  "#94a3b8"],
                ],
                showscale=False,
                text=[[year_tops.get(yr, "") if year_tops.get(yr) == etf else ""
                       for yr in years_sorted]
                      for etf in heat_etfs],
                texttemplate="%{text}",
                textfont={"size": 9, "color": "#1e293b"},
            ))
            fig_heat.update_layout(
                height=280,
                **CHART_LAYOUT,
                title="Top Pick per Training Window",
                xaxis=dict(title="Training Start Year", tickangle=0,
                           gridcolor="#e2e8f0"),
                yaxis=dict(title="", gridcolor="#e2e8f0"),
                margin=dict(l=60, r=0, t=40, b=40),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Conviction score breakdown bar chart (WHITE background) ────────
        st.subheader("📐 Conviction Score Decomposition")
        st.caption(
            "How vote share, OOS directional accuracy and Hurst H "
            "contribute to each ETF's conviction score"
        )

        etf_list    = conviction_df["etf"].tolist()
        vote_vals   = (conviction_df["vote_share"]   * CONVICTION_W["votes"]).tolist()
        diracc_vals = (conviction_df["norm_dir_acc"] * CONVICTION_W["dir_acc"]).tolist()
        hurst_vals  = (conviction_df["norm_H"]       * CONVICTION_W["hurst"]).tolist()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name=f"Vote Share ({int(CONVICTION_W['votes']*100)}%)",
            x=etf_list, y=vote_vals, marker_color="#0284c7"))
        fig_bar.add_trace(go.Bar(
            name=f"OOS Dir Accuracy ({int(CONVICTION_W['dir_acc']*100)}%)",
            x=etf_list, y=diracc_vals, marker_color="#d97706"))
        fig_bar.add_trace(go.Bar(
            name=f"Avg Hurst H ({int(CONVICTION_W['hurst']*100)}%)",
            x=etf_list, y=hurst_vals, marker_color="#10b981"))

        fig_bar.update_layout(
            barmode="stack",
            height=380,
            **CHART_LAYOUT,
            title="Conviction = Votes×0.35 + Dir Accuracy×0.40 + Hurst H×0.25",
            yaxis=dict(title="Score Contribution", gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Per-year flat detail table ─────────────────────────────────────
        if flat_df is not None and len(flat_df) > 0:
            with st.expander("🔍 Per-Year Per-ETF Detail Table", expanded=False):
                show_cols = [c for c in
                             ["year", "etf", "pred_ret_pct", "dir_acc", "H", "mae", "rmse", "model"]
                             if c in flat_df.columns]
                flat_show = flat_df[show_cols].copy()
                col_rename = {
                    "year":         "Year",
                    "etf":          "ETF",
                    "pred_ret_pct": "Pred Ret %",
                    "dir_acc":      "Dir Acc % ★",
                    "H":            "Hurst H",
                    "mae":          "MAE",
                    "rmse":         "RMSE",
                    "model":        "Model",
                }
                flat_show = flat_show.rename(columns=col_rename)
                st.caption(
                    "★ Dir Acc % = OOS directional accuracy: "
                    "% of test-period days where predicted return direction matched actual. "
                    "This drives 40% of the conviction score."
                )
                st.dataframe(
                    flat_show.sort_values(["Year", "Dir Acc % ★"], ascending=[True, False]),
                    use_container_width=True,
                    hide_index=True,
                )

        st.info(
            "💡 **Tip:** The sweep runs automatically at 8 PM EST on weekdays via GitHub Actions. "
            "To trigger manually: GitHub repo → Actions → *Consensus Sweep* → *Run workflow*. "
            "Only the latest run is kept on HuggingFace — previous stamped files are deleted automatically."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OOS CUMULATIVE RETURN CHART
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("OOS Period: Cumulative Return — Model Signal vs Benchmark")
    st.caption("Cumulative actual return of the model's top-ranked ETF each day vs the selected benchmark.")

    bench_choice = benchmark if benchmark != "None" else "SPY"

    if audit_df is not None and len(audit_df) > 0 and ret_df is not None:

        oos = audit_df.copy()
        oos["date"] = pd.to_datetime(oos["date"])
        oos = oos.sort_values("date")

        if rankings is not None:
            rank1 = (rankings[rankings["rank"] == 1][["date", "etf"]]
                     .rename(columns={"etf": "top_etf"}))
            rank1["date"] = pd.to_datetime(rank1["date"])
        else:
            rank1 = (oos.groupby("date")
                     .apply(lambda g: g.loc[g["predicted_ret_pct"].idxmax(), "signal_etf"]
                            if "predicted_ret_pct" in g.columns else g.iloc[0]["signal_etf"])
                     .reset_index().rename(columns={0: "top_etf"}))

        signal_rets = []
        for _, row in rank1.iterrows():
            d   = row["date"]
            etf = row["top_etf"]
            if etf in ret_df.columns and d in ret_df.index:
                signal_rets.append({"date": d, "signal_ret": float(ret_df.loc[d, etf])})
            else:
                signal_rets.append({"date": d, "signal_ret": np.nan})

        signal_df = pd.DataFrame(signal_rets).dropna().set_index("date").sort_index()

        bench_rets = None
        if bench_choice != "None" and bench_ret is not None and bench_choice in bench_ret.columns:
            bench_rets = bench_ret[bench_choice].reindex(signal_df.index).dropna()

        if len(signal_df) >= 2:
            cum_signal = (1 + signal_df["signal_ret"]).cumprod() - 1

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_signal.index, y=cum_signal.values * 100,
                mode="lines", name="Model Signal",
                line=dict(color="#0ea5e9", width=2.5),
            ))

            if bench_rets is not None and len(bench_rets) >= 2:
                cum_bench = (1 + bench_rets).cumprod() - 1
                fig.add_trace(go.Scatter(
                    x=cum_bench.index, y=cum_bench.values * 100,
                    mode="lines", name=bench_choice,
                    line=dict(color=ETF_COLORS.get(bench_choice, "#6366f1"),
                              width=1.8, dash="dot"),
                ))

            fig.update_layout(
                height=440,
                **CHART_LAYOUT,
                title="Cumulative Return — OOS Period (%)",
                xaxis=dict(gridcolor="#e2e8f0", title="Date"),
                yaxis=dict(gridcolor="#e2e8f0", title="Cumulative Return (%)"),
                legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
            st.plotly_chart(fig, use_container_width=True)

            total_ret = float(cum_signal.iloc[-1]) * 100
            n_days    = len(signal_df)
            ann_ret   = ((1 + total_ret / 100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0
            daily_std = float(signal_df["signal_ret"].std()) * np.sqrt(252) * 100
            sharpe    = ann_ret / daily_std if daily_std > 0 else 0

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total OOS Return", f"{total_ret:+.2f}%")
            sc2.metric("Ann. Return (est.)", f"{ann_ret:+.2f}%")
            sc3.metric("Ann. Volatility", f"{daily_std:.2f}%")
            sc4.metric("Sharpe (est.)", f"{sharpe:.2f}")
        else:
            st.info("Not enough OOS data yet to plot.")
    else:
        st.info("OOS chart will appear once the training pipeline has run and the audit trail is populated.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 OOS Model Performance")

    def compute_performance(audit, bench_ret_df, bench_col, rankings_df):
        if audit is None or rankings_df is None or bench_ret_df is None:
            return None
        r1 = (rankings_df[rankings_df["rank"] == 1][["date", "etf"]]
              .rename(columns={"etf": "top_etf"}))
        r1["date"] = pd.to_datetime(r1["date"])
        rows = []
        for _, row in r1.iterrows():
            d, etf = row["date"], row["top_etf"]
            if etf in ret_df.columns and d in ret_df.index:
                rows.append({"date": d, "ret": float(ret_df.loc[d, etf]), "etf": etf})
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("date").sort_index()
        df["cum"] = (1 + df["ret"]).cumprod()
        n = len(df)
        if n < 2:
            return None
        total_ret  = float(df["cum"].iloc[-1]) - 1
        ann_ret    = ((1 + total_ret) ** (252 / n) - 1) * 100
        ann_vol    = df["ret"].std() * np.sqrt(252) * 100
        sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0
        roll_max   = df["cum"].cummax()
        drawdowns  = (df["cum"] - roll_max) / roll_max
        max_dd_pt  = float(drawdowns.min()) * 100
        max_dd_day = float(df["ret"].min()) * 100
        last15     = df.tail(15)
        hit_ratio  = (last15["ret"] > 0).sum() / len(last15) * 100 if len(last15) > 0 else 0
        bench_stats = {}
        if bench_col in bench_ret_df.columns:
            b = bench_ret_df[bench_col].reindex(df.index).dropna()
            if len(b) >= 2:
                b_cum    = (1 + b).cumprod()
                b_total  = float(b_cum.iloc[-1]) - 1
                b_ann    = ((1 + b_total) ** (252 / len(b)) - 1) * 100
                b_vol    = b.std() * np.sqrt(252) * 100
                b_sharpe = b_ann / b_vol if b_vol > 0 else 0
                b_dd     = ((b_cum - b_cum.cummax()) / b_cum.cummax()).min() * 100
                bench_stats = {"bench_ann_ret": b_ann, "bench_sharpe": b_sharpe,
                               "bench_max_dd": float(b_dd)}
        return {"ann_ret": ann_ret, "sharpe": sharpe, "max_dd_pt": max_dd_pt,
                "max_dd_day": max_dd_day, "hit_ratio": hit_ratio, "n_days": n,
                **bench_stats}

    bench_col = benchmark if benchmark != "None" else "SPY"

    if (audit_df is not None and rankings is not None
            and ret_df is not None and bench_ret is not None):
        perf = compute_performance(audit_df, bench_ret, bench_col, rankings)
        if perf:
            st.caption(f"OOS period: {perf['n_days']} trading days · Benchmark: {bench_col}")
            m1, m2, m3, m4, m5 = st.columns(5)
            bench_ann = perf.get("bench_ann_ret")
            bench_sh  = perf.get("bench_sharpe")
            bench_dd  = perf.get("bench_max_dd")
            m1.metric("Ann. Return (OOS)", f"{perf['ann_ret']:+.2f}%",
                      delta=f"vs {bench_col}: {perf['ann_ret'] - bench_ann:+.2f}%" if bench_ann else None)
            m2.metric("Sharpe Ratio", f"{perf['sharpe']:.2f}",
                      delta=f"vs {bench_col}: {perf['sharpe'] - bench_sh:+.2f}" if bench_sh else None)
            m3.metric("Max DD (Peak→Trough)", f"{perf['max_dd_pt']:.2f}%",
                      delta=f"vs {bench_col}: {perf['max_dd_pt'] - bench_dd:+.2f}%" if bench_dd else None,
                      delta_color="inverse")
            m4.metric("Max DD (Worst Day)", f"{perf['max_dd_day']:.2f}%", delta_color="inverse")
            m5.metric(f"Hit Ratio (Last 15d)", f"{perf['hit_ratio']:.1f}%",
                      delta="Above 50% = positive signal" if perf['hit_ratio'] > 50 else "Below 50%",
                      delta_color="normal" if perf['hit_ratio'] > 50 else "inverse")
            st.divider()
            if bench_ann is not None:
                comp_data = {
                    "Metric": ["Ann. Return", "Sharpe Ratio", "Max DD (P→T)",
                               "Max DD (Day)", "Hit Ratio (15d)"],
                    "Model Signal": [f"{perf['ann_ret']:+.2f}%", f"{perf['sharpe']:.2f}",
                                     f"{perf['max_dd_pt']:.2f}%", f"{perf['max_dd_day']:.2f}%",
                                     f"{perf['hit_ratio']:.1f}%"],
                    bench_col: [f"{bench_ann:+.2f}%", f"{bench_sh:.2f}",
                                f"{bench_dd:.2f}%", "—", "—"],
                }
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        else:
            st.info("Performance stats will populate once OOS actual returns are available.")
    else:
        st.info("No performance data yet. Run the training pipeline for at least 2 days.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HURST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🌊 Hurst Exponent — Model Selection Logic")
    st.markdown("""
    The **Hurst Exponent (H)** via R/S analysis determines which model is used per ETF:
    - **H > 0.52** → Long-term memory → **ARMA-RNN-LSTM Hybrid** (paper §3)
    - **H ≈ 0.50** → Near random walk → **Standalone RNN** (paper §4.3)
    - **H < 0.45** → Anti-persistent → **Standalone RNN**
    """)

    if metrics_df is not None and len(metrics_df) > 0:
        latest = metrics_df.sort_values("run_date").groupby("etf").last().reset_index()
        fig_h  = go.Figure()
        for _, row in latest.iterrows():
            color = "#0ea5e9" if row["hurst_H"] > HURST_THRESH else "#f59e0b"
            fig_h.add_trace(go.Bar(
                x=[row["etf"]], y=[row["hurst_H"]],
                name=row["etf"], marker_color=color,
                text=f"H={row['hurst_H']:.3f}", textposition="outside",
            ))
        fig_h.add_hline(y=HURST_THRESH, line_dash="dash", line_color="#ef4444",
                        annotation_text=f"Long-memory threshold (H={HURST_THRESH})",
                        annotation_position="top right")
        fig_h.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8",
                        annotation_text="Random walk (H=0.5)")
        fig_h.update_layout(
            height=380,
            **CHART_LAYOUT,
            title="Hurst Exponent by ETF (latest training run)",
            yaxis=dict(range=[0.3, 0.75], gridcolor="#e2e8f0", title="H"),
            xaxis=dict(gridcolor="#e2e8f0"),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_h, use_container_width=True)
        hurst_tbl = latest[["etf", "hurst_H", "memory_type", "model_used"]].copy()
        hurst_tbl.columns = ["ETF", "Hurst H", "Memory Type", "Model Used"]
        st.dataframe(hurst_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("Hurst analysis will appear after the first training run.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Audit Trail — Last 15 Trading Days")
    st.caption("One row per day · Top pick ETF · Actual return from source data · N/A = today")

    if rankings is not None and len(rankings) > 0 and ret_df is not None:
        today = pd.Timestamp(datetime.now().date())
        rank1 = (rankings[rankings["rank"] == 1]
                 [["date", "etf", "predicted_return_pct",
                   "hurst_H", "model_used", "direction_accuracy",
                   "current_price", "predicted_price"]]
                 .copy())
        rank1["date"] = pd.to_datetime(rank1["date"])
        rank1 = rank1.sort_values("date", ascending=False).head(AUDIT_ROWS)

        audit_lookup = {}
        if audit_df is not None:
            af = audit_df.copy()
            af["date"] = pd.to_datetime(af["date"])
            af_filled = af[af["actual_ret_pct"].notna()]
            for _, ar in af_filled.iterrows():
                audit_lookup[(ar["date"], ar["signal_etf"])] = float(ar["actual_ret_pct"])

        display_rows = []
        for _, row in rank1.iterrows():
            d   = row["date"]
            etf = row["etf"]
            if d >= today:
                actual_ret = None
            elif etf in ret_df.columns and d in ret_df.index:
                actual_ret = round(float(ret_df.loc[d, etf]) * 100, 4)
            elif (d, etf) in audit_lookup:
                actual_ret = round(audit_lookup[(d, etf)], 4)
            else:
                actual_ret = None
            pred = row["predicted_return_pct"]
            result = ("✅" if actual_ret is not None and actual_ret > 0
                      else ("❌" if actual_ret is not None and actual_ret <= 0 else "⏳"))
            dir_correct = "—"
            if actual_ret is not None and pd.notna(pred):
                dir_correct = "✅" if (pred > 0) == (actual_ret > 0) else "❌"
            display_rows.append({
                "Date":        d.strftime("%Y-%m-%d"),
                "Top Pick":    etf,
                "Pred Ret%":   f"{pred:+.3f}%" if pd.notna(pred) else "—",
                "Actual Ret%": f"{actual_ret:+.3f}%" if actual_ret is not None else "N/A",
                "Return":      result,
                "Direction":   dir_correct,
                "Hurst H":     round(row["hurst_H"], 3),
                "Model":       row["model_used"],
                "Dir Acc%":    round(row["direction_accuracy"], 1),
            })

        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Date":        st.column_config.TextColumn("Date"),
                         "Top Pick":    st.column_config.TextColumn("Top Pick"),
                         "Pred Ret%":   st.column_config.TextColumn("Pred Ret%"),
                         "Actual Ret%": st.column_config.TextColumn("Actual Ret%"),
                         "Return":      st.column_config.TextColumn("Return", width="small"),
                         "Direction":   st.column_config.TextColumn("Dir OK?", width="small"),
                         "Hurst H":     st.column_config.NumberColumn("Hurst H", format="%.3f"),
                         "Model":       st.column_config.TextColumn("Model"),
                         "Dir Acc%":    st.column_config.NumberColumn("OOS Acc%", format="%.1f"),
                     })

        known = [r for r in display_rows if r["Actual Ret%"] != "N/A"]
        if known:
            wins    = sum(1 for r in known if float(r["Actual Ret%"].replace("%", "")) > 0)
            total   = len(known)
            avg_ret = np.mean([float(r["Actual Ret%"].replace("%", "")) for r in known])
            st.divider()
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Hit Ratio", f"{wins/total*100:.1f}%")
            wc2.metric("Days Evaluated", total)
            wc3.metric("Avg Actual Return", f"{avg_ret:+.3f}%")
    else:
        st.info("Audit trail will populate after the first training run.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("ℹ️ About the ARMA-RNN-LSTM Model")
    st.markdown("""
    ### 📄 Paper Reference
    **Xiao H (2025)** — *Enhanced separation of long-term memory from short-term memory
    on top of LSTM: Neural network-based stock index forecasting.*
    PLoS ONE 20(6): e0322737. https://doi.org/10.1371/journal.pone.0322737

    ---

    ### 🔬 Core Insight
    Standard LSTMs are supposed to separate long-term from short-term memory internally.
    The paper shows they often **misclassify short-term memory as long-term**, reducing accuracy.
    The fix: **pre-separate the two memory types externally** before feeding into LSTM.

    ---

    ### 🏗️ 3-Stage Pipeline

    | Stage | Model | Role |
    |-------|-------|------|
    | **1** | SimpleRNN | Captures **short-term memory** |
    | **2** | ResidualLSTM | Trained on RNN residuals → extracts **long-term memory** |
    | **3** | HybridLSTM | Fuses both → final refined forecast |

    ---

    ### 🌊 Hurst Decision Rule
    - **H > 0.52** → Use ARMA-RNN-LSTM hybrid
    - **H ≈ 0.50** → Use RNN only

    ---

    ### 🔁 Consensus Sweep (v2)
    Trains all 17 training-start year windows (2008–2024) in parallel and scores
    conviction using:

    | Component | Weight | What it measures |
    |-----------|--------|-----------------|
    | Vote share | **35%** | How many windows agree on the same top ETF |
    | OOS Dir Accuracy | **40%** | % of test-period days where predicted direction matched actual |
    | Avg Hurst H | **25%** | Trend-persistence signal strength |

    Avg predicted return is shown for reference but **not used in scoring** — it has no
    ground-truth anchor. OOS directional accuracy is the genuine predicted-vs-actual
    sync measure.

    ---

    ### ⚙️ Fixed Parameters (paper-optimal)
    | Parameter | Value |
    |-----------|-------|
    | Train/Test split | **80/20** |
    | Hurst threshold | **0.52** |
    | Lookback window | **10 days** |

    ---

    ### ⚠️ Disclaimer
    Educational and research purposes only. Not financial advice.
    """)
