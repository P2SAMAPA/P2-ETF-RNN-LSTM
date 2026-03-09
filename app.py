# app.py
# Streamlit UI for P2-ETF-RNN-LSTM
# Live inference + results dashboard
# UI inspired by the P2-ETF Regime-Aware Rotation Model design

import os
import warnings
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF-RNN-LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_RESULTS  = "P2SAMAPA/p2-etf-rnn-lstm-results"
HF_SOURCE   = "P2SAMAPA/p2-etf-deepwave-dl"
TARGET_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCH       = ["SPY", "AGG"]
ETF_LABELS  = {
    "TLT": "iShares 20yr Treasury",
    "LQD": "iShares Inv Grade Corp",
    "HYG": "iShares High Yield Bond",
    "VNQ": "Vanguard Real Estate",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
}
ETF_COLORS  = {
    "TLT": "#00d4ff", "LQD": "#f472b6", "HYG": "#10b981",
    "VNQ": "#7c3aed", "GLD": "#f59e0b", "SLV": "#94a3b8",
    "SPY": "#6366f1", "AGG": "#84cc16",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.title("P2-ETF RNN-LSTM")
    st.caption("ARMA-RNN-LSTM Hybrid Neural Engine")
    st.caption(f"🕐 EST: {datetime.now().strftime('%a %b %d, %H:%M')}")
    st.divider()

    st.subheader("⚙️ Configuration")
    lookback_years = st.slider("Training Lookback (Years)", 2, 16, 8,
                                help="Years of history used for model training")
    train_split = st.slider("Train/Test Split (%)", 50, 80, 63,
                             help="% of data used for training (paper: 62.5%)")
    st.divider()

    st.subheader("🎯 Model Settings")
    hurst_threshold = st.slider("Long-Memory Threshold (H)", 0.50, 0.60, 0.52, 0.01,
                                 help="H above this → ARMA-RNN-LSTM hybrid; else RNN only")
    show_rnn_baseline = st.checkbox("Show RNN baseline comparison", True)
    show_combined     = st.checkbox("Show RNN+LSTM (no hybrid) comparison", True)
    st.divider()

    st.subheader("📊 Display")
    benchmark = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    n_audit_rows = st.slider("Audit Trail Rows", 10, 90, 30)
    st.divider()

    force_refresh = st.button("🔄 Force Data Refresh", use_container_width=True)
    run_inference = st.button("🚀 Run Live Inference", use_container_width=True,
                               type="primary")


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_results_parquet(filename: str) -> pd.DataFrame | None:
    token = os.environ.get("HF_TOKEN", "")
    try:
        local = hf_hub_download(repo_id=HF_RESULTS, filename=filename,
                                repo_type="dataset", token=token or None)
        df = pd.read_parquet(local)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "run_date" in df.columns:
            df["run_date"] = pd.to_datetime(df["run_date"])
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


def clear_cache():
    load_results_parquet.clear()
    load_source_parquet.clear()


if force_refresh:
    clear_cache()
    st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading latest results from HuggingFace…"):
    predictions  = load_results_parquet("predictions.parquet")
    rankings     = load_results_parquet("rankings.parquet")
    metrics_df   = load_results_parquet("metrics.parquet")
    audit_df     = load_results_parquet("audit_trail.parquet")
    price_df     = load_source_parquet("data/etf_price.parquet")
    ret_df       = load_source_parquet("data/etf_ret.parquet")
    bench_price  = load_source_parquet("data/bench_price.parquet")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 📈 P2-ETF RNN-LSTM Neural Forecasting Engine")
st.caption(
    "ARMA-RNN-LSTM Hybrid Model · Xiao (2025) PLoS ONE · "
    "ETFs: " + " · ".join(TARGET_ETFS)
)

# ── Data status badges ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
if price_df is not None:
    col1.success(
        f"✅ Dataset: {len(price_df):,} rows × {len(price_df.columns)} cols "
        f"({price_df.index[0].date()} → {price_df.index[-1].date()})"
    )
else:
    col1.error("❌ Source data unavailable")

if predictions is not None:
    col2.success(f"✅ Predictions loaded ({len(predictions):,} rows)")
else:
    col2.warning("⚠️ No predictions yet — run training pipeline")

if rankings is not None:
    latest_date = rankings["date"].max().strftime("%Y-%m-%d")
    col3.success(f"✅ Rankings available — latest: {latest_date}")
else:
    col3.warning("⚠️ No rankings yet")

if metrics_df is not None:
    col4.success(f"✅ Metrics loaded ({len(metrics_df):,} runs)")
else:
    col4.info("ℹ️ No metrics yet")

st.divider()

# ── Live Inference ─────────────────────────────────────────────────────────────
if run_inference:
    st.subheader("🚀 Live Inference")
    _run_live_inference(price_df, ret_df, hurst_threshold, train_split / 100)

# ── Top Signal Banner ──────────────────────────────────────────────────────────
if rankings is not None and len(rankings) > 0:
    latest_rankings = rankings[rankings["date"] == rankings["date"].max()].sort_values("rank")

    if len(latest_rankings) > 0:
        top = latest_rankings.iloc[0]
        ret = top["predicted_return_pct"]
        signal_color = "#1a4a2e" if ret >= 0 else "#4a1a1a"
        signal_border = "#10b981" if ret >= 0 else "#ef4444"

        st.markdown(f"""
        <div style="background:{signal_color}; border:1px solid {signal_border};
                    border-radius:10px; padding:20px 28px; margin-bottom:16px;">
            <div style="font-size:11px; letter-spacing:3px; color:#94a3b8; margin-bottom:6px;">
                NEXT TRADING DAY SIGNAL — {top['date'].strftime('%A %b %d, %Y')}
                <span style="float:right; font-size:10px;">MODEL: {top['model_used']}</span>
            </div>
            <div style="display:flex; align-items:center; gap:32px;">
                <div>
                    <div style="font-size:48px; font-weight:900; color:white; line-height:1;">
                        {top['etf']}
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:4px;">
                        {ETF_LABELS.get(top['etf'], '')}
                    </div>
                    <div style="margin-top:8px; font-size:13px; color:#94a3b8;">
                        Predicted Return:
                        <span style="color:{'#10b981' if ret >= 0 else '#ef4444'}; font-weight:700;">
                            {'+' if ret >= 0 else ''}{ret:.3f}%
                        </span>
                        &nbsp;|&nbsp; Dir Accuracy:
                        <span style="color:#f59e0b; font-weight:700;">
                            {top['direction_accuracy']:.1f}%
                        </span>
                        &nbsp;|&nbsp; H =
                        <span style="color:#00d4ff; font-weight:700;">{top['hurst_H']:.3f}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── All ETF predictions for latest date ───────────────────────────────
        st.subheader("📊 All ETF Predictions — Next Trading Day")
        pred_cols = st.columns(3)
        for i, row in latest_rankings.iterrows():
            col_idx = int(row["rank"] - 1) % 3
            etf     = row["etf"]
            r       = row["predicted_return_pct"]
            color   = "#10b981" if r > 0 else "#ef4444"
            with pred_cols[col_idx]:
                st.markdown(f"""
                <div style="border:1px solid #1f2d45; border-radius:8px;
                            padding:14px; margin-bottom:10px; background:#111827;">
                    <div style="font-size:10px; color:#64748b; margin-bottom:4px;">
                        #{int(row['rank'])} · {ETF_LABELS.get(etf, etf)}
                    </div>
                    <div style="font-size:22px; font-weight:800;
                                color:{ETF_COLORS.get(etf, '#fff')};">{etf}</div>
                    <div style="font-size:20px; font-weight:700; color:{color}; margin:4px 0;">
                        {'+' if r > 0 else ''}{r:.3f}%
                    </div>
                    <div style="font-size:11px; color:#64748b;">
                        H={row['hurst_H']:.3f} · {row['model_used']} · 
                        Dir Acc {row['direction_accuracy']:.1f}%
                    </div>
                    <div style="font-size:10px; color:#94a3b8; margin-top:4px;">
                        ${row['current_price']:.2f} → ${row['predicted_price']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Charts",
    "📊 Model Performance",
    "🌊 Hurst Analysis",
    "📋 Audit Trail",
    "ℹ️ About the Model",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Price History & Model Forecasts")

    etf_sel = st.selectbox("Select ETF", TARGET_ETFS,
                            format_func=lambda x: f"{x} — {ETF_LABELS[x]}")

    if price_df is not None and etf_sel in price_df.columns:
        px_series = price_df[etf_sel].dropna()
        years_back = lookback_years
        cutoff = px_series.index[-1] - pd.DateOffset(years=years_back)
        px_plot = px_series[px_series.index >= cutoff]

        # Benchmark overlay
        bench_series = None
        if benchmark != "None" and bench_price is not None and benchmark in bench_price.columns:
            bench_series = bench_price[benchmark].dropna()
            bench_series = bench_series[bench_series.index >= cutoff]

        fig = go.Figure()

        # ETF price line
        fig.add_trace(go.Scatter(
            x=px_plot.index, y=px_plot.values,
            mode="lines", name=etf_sel,
            line=dict(color=ETF_COLORS.get(etf_sel, "#00d4ff"), width=2),
        ))

        # Benchmark (secondary axis)
        if bench_series is not None and len(bench_series) > 0:
            fig.add_trace(go.Scatter(
                x=bench_series.index, y=bench_series.values,
                mode="lines", name=benchmark,
                line=dict(color=ETF_COLORS.get(benchmark, "#6366f1"),
                          width=1.5, dash="dot"),
                yaxis="y2",
            ))

        # Prediction markers from predictions dataset
        if predictions is not None:
            etf_preds = predictions[predictions["etf"] == etf_sel].copy()
            etf_preds = etf_preds[etf_preds["date"] >= cutoff]
            if len(etf_preds) > 0:
                colors = ["#10b981" if r > 0 else "#ef4444"
                          for r in etf_preds["predicted_return_pct"]]
                fig.add_trace(go.Scatter(
                    x=etf_preds["date"],
                    y=etf_preds["predicted_price"],
                    mode="markers",
                    name="Predicted Price",
                    marker=dict(color=colors, size=6, symbol="circle"),
                ))

        fig.update_layout(
            height=420,
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#0a0e1a",
            showlegend=True,
            legend=dict(bgcolor="#111827", bordercolor="#1f2d45"),
            xaxis=dict(gridcolor="#1f2d45", showgrid=True),
            yaxis=dict(gridcolor="#1f2d45", showgrid=True, title=f"{etf_sel} Price"),
            yaxis2=dict(overlaying="y", side="right",
                        title=benchmark if benchmark != "None" else ""),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        if ret_df is not None and etf_sel in ret_df.columns:
            ret_series = ret_df[etf_sel].dropna()
            ret_cut    = ret_series[ret_series.index >= cutoff]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=ret_cut.values * 100,
                nbinsx=80,
                name="Daily Returns",
                marker_color=ETF_COLORS.get(etf_sel, "#00d4ff"),
                opacity=0.75,
            ))
            fig2.update_layout(
                height=220,
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#0a0e1a",
                title=f"{etf_sel} Return Distribution (%)",
                xaxis_title="Daily Return (%)",
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Price data not available. Check HuggingFace connection.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Metrics")

    if metrics_df is not None and len(metrics_df) > 0:
        latest_metrics = (
            metrics_df.sort_values("run_date")
            .groupby("etf").last().reset_index()
        )

        # Metric cards
        metric_cols = st.columns(len(TARGET_ETFS))
        for i, etf in enumerate(TARGET_ETFS):
            row = latest_metrics[latest_metrics["etf"] == etf]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            with metric_cols[i]:
                dir_acc = row.get("hybrid_dir_acc", row.get("rnn_dir_acc", 0))
                mae     = row.get("hybrid_mae",     row.get("rnn_mae", 0))
                st.metric(
                    label=etf,
                    value=f"{dir_acc:.1f}%",
                    delta=f"MAE {mae:.5f}",
                    help=f"Model: {row['model_used']} | H={row['hurst_H']:.3f}",
                )

        st.divider()

        # Direction accuracy bar chart
        if show_rnn_baseline:
            fig = go.Figure()
            for col, label, color in [
                ("hybrid_dir_acc",   "ARMA-RNN-LSTM", "#00d4ff"),
                ("rnn_dir_acc",      "RNN baseline",  "#7c3aed"),
                ("combined_dir_acc", "RNN+LSTM",      "#f59e0b"),
            ]:
                if col in latest_metrics.columns:
                    fig.add_trace(go.Bar(
                        x=latest_metrics["etf"],
                        y=latest_metrics[col],
                        name=label,
                        marker_color=color,
                    ))
        else:
            fig = go.Figure(go.Bar(
                x=latest_metrics["etf"],
                y=latest_metrics.get("hybrid_dir_acc",
                                      latest_metrics.get("rnn_dir_acc", [])),
                marker_color=[ETF_COLORS.get(e, "#00d4ff")
                              for e in latest_metrics["etf"]],
            ))

        fig.update_layout(
            height=350,
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#0a0e1a",
            title="Directional Accuracy (%) by ETF & Model",
            yaxis=dict(range=[40, 80], gridcolor="#1f2d45", title="%"),
            xaxis=dict(gridcolor="#1f2d45"),
            barmode="group",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor="#111827"),
        )
        # 50% line
        fig.add_hline(y=50, line_dash="dot", line_color="#ef4444",
                      annotation_text="Random (50%)")
        st.plotly_chart(fig, use_container_width=True)

        # MAE comparison
        st.subheader("MAE Comparison — Hybrid vs Baselines")
        mae_cols = [c for c in latest_metrics.columns if "mae" in c.lower()]
        if mae_cols:
            fig3 = go.Figure()
            colors_mae = {"hybrid_mae": "#00d4ff", "rnn_mae": "#7c3aed",
                          "combined_mae": "#f59e0b"}
            labels_mae = {"hybrid_mae": "ARMA-RNN-LSTM",
                          "rnn_mae": "RNN baseline",
                          "combined_mae": "RNN+LSTM"}
            for col in mae_cols:
                if col in latest_metrics.columns:
                    fig3.add_trace(go.Bar(
                        x=latest_metrics["etf"],
                        y=latest_metrics[col],
                        name=labels_mae.get(col, col),
                        marker_color=colors_mae.get(col, "#94a3b8"),
                    ))
            fig3.update_layout(
                height=300, template="plotly_dark",
                paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
                barmode="group", margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="#111827"),
                yaxis=dict(gridcolor="#1f2d45", title="MAE (log-return)"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Metrics table
        st.subheader("Full Metrics Table")
        display_cols = ["etf", "model_used", "hurst_H", "memory_type",
                        "hybrid_dir_acc", "hybrid_mae", "hybrid_rmse",
                        "rnn_dir_acc", "rnn_mae", "train_samples", "test_samples"]
        show_cols = [c for c in display_cols if c in latest_metrics.columns]
        st.dataframe(latest_metrics[show_cols], use_container_width=True)
    else:
        st.info("No metrics available yet. Run the training pipeline first.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HURST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🌊 Hurst Exponent Analysis")
    st.markdown("""
    The **Hurst Exponent (H)** via R/S analysis determines which model to use:
    - **H > 0.52** → Long-term memory detected → **ARMA-RNN-LSTM Hybrid** (paper §3)
    - **H ≈ 0.5**  → Near random walk → **Standalone RNN** (paper §4.3)
    - **H < 0.45** → Anti-persistent → **Standalone RNN**
    """)

    if metrics_df is not None:
        latest = metrics_df.sort_values("run_date").groupby("etf").last().reset_index()

        fig_h = go.Figure()
        for _, row in latest.iterrows():
            color = "#00d4ff" if row["hurst_H"] > hurst_threshold else "#f59e0b"
            fig_h.add_trace(go.Bar(
                x=[row["etf"]],
                y=[row["hurst_H"]],
                name=row["etf"],
                marker_color=color,
                text=f"H={row['hurst_H']:.3f}",
                textposition="outside",
            ))

        fig_h.add_hline(y=hurst_threshold, line_dash="dash",
                        line_color="#ef4444",
                        annotation_text=f"Long-memory threshold (H={hurst_threshold})",
                        annotation_position="top right")
        fig_h.add_hline(y=0.5, line_dash="dot", line_color="#64748b",
                        annotation_text="Random walk (H=0.5)")

        fig_h.update_layout(
            height=380, template="plotly_dark",
            paper_bgcolor="#111827", plot_bgcolor="#0a0e1a",
            title="Hurst Exponent by ETF (latest training run)",
            yaxis=dict(range=[0.3, 0.9], gridcolor="#1f2d45",
                       title="Hurst Exponent H"),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Table
        hurst_table = latest[["etf", "hurst_H", "memory_type", "model_used"]].copy()
        hurst_table.columns = ["ETF", "Hurst H", "Memory Type", "Model Used"]
        st.dataframe(hurst_table, use_container_width=True, hide_index=True)

    else:
        # Show theoretical explanation if no data yet
        st.info("Hurst analysis will appear after the first training run.")
        st.markdown("""
        **Expected results based on literature:**
        | ETF | Asset Class | Expected H | Notes |
        |-----|-------------|------------|-------|
        | GLD | Gold | ~0.60–0.65 | Strong trend persistence |
        | TLT | Long Treasury | ~0.58–0.62 | Rate cycle persistence |
        | VNQ | Real Estate | ~0.55–0.60 | Property cycle memory |
        | HYG | High Yield | ~0.52–0.58 | Credit cycle |
        | LQD | IG Corp Bond | ~0.50–0.55 | Near random |
        | SLV | Silver | ~0.50–0.55 | More volatile |
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUDIT TRAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"📋 Audit Trail — Last {n_audit_rows} Trading Days")

    if audit_df is not None and len(audit_df) > 0:
        audit_show = (
            audit_df.sort_values("date", ascending=False)
            .head(n_audit_rows)
            .copy()
        )
        audit_show["date"] = audit_show["date"].dt.strftime("%Y-%m-%d")

        # Format return columns
        for col in ["predicted_ret_pct", "actual_ret_pct"]:
            if col in audit_show.columns:
                audit_show[col] = audit_show[col].apply(
                    lambda x: f"{x:+.3f}%" if pd.notna(x) else "—"
                )

        # Highlight top pick column
        st.dataframe(
            audit_show,
            use_container_width=True,
            column_config={
                "date":              st.column_config.TextColumn("Date"),
                "signal_etf":        st.column_config.TextColumn("Signal ETF"),
                "predicted_ret_pct": st.column_config.TextColumn("Predicted Ret%"),
                "actual_ret_pct":    st.column_config.TextColumn("Actual Ret%"),
                "hurst_H":           st.column_config.NumberColumn("Hurst H", format="%.3f"),
                "model_used":        st.column_config.TextColumn("Model"),
                "direction_accuracy":st.column_config.NumberColumn("Dir Acc%", format="%.1f"),
            },
        )

        # Win rate summary
        if "actual_ret_pct" in audit_df.columns:
            filled = audit_df[audit_df["actual_ret_pct"].notna()].copy()
            if len(filled) > 0:
                filled["actual_ret_pct"] = pd.to_numeric(
                    filled["actual_ret_pct"].astype(str).str.replace("%",""),
                    errors="coerce"
                )
                wins    = (filled["actual_ret_pct"] > 0).sum()
                total   = len(filled)
                win_pct = wins / total * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Win Rate", f"{win_pct:.1f}%")
                c2.metric("Total Signals", total)
                c3.metric("Wins", wins)
                avg_ret = filled["actual_ret_pct"].mean()
                c4.metric("Avg Actual Return", f"{avg_ret:+.3f}%")
    else:
        st.info("Audit trail will populate after the first training run.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("ℹ️ About the ARMA-RNN-LSTM Model")
    st.markdown("""
    ### 📄 Paper Reference
    **Xiao H (2025)** — *Enhanced separation of long-term memory from short-term memory
    on top of LSTM: Neural network-based stock index forecasting.*
    PLoS ONE 20(6): e0322737. https://doi.org/10.1371/journal.pone.0322737

    ---

    ### 🔬 Core Insight
    Standard LSTM networks are supposed to separate long-term from short-term memory
    internally via their gating mechanism. However, the paper demonstrates that **LSTMs
    often misclassify short-term memory as long-term**, reducing forecast accuracy.

    The solution: **pre-separate the two memory types externally** before feeding into LSTM.

    ---

    ### 🏗️ 3-Stage Pipeline

    | Stage | Model | Role | Equation |
    |-------|-------|------|----------|
    | **1** | SimpleRNN | Captures **short-term memory** (by design, RNNs can't do long-term) | Eq. 24–27 |
    | **2** | ResidualLSTM | Trained on RNN residuals → extracts **long-term memory** | Eq. 28–29 |
    | **3** | HybridLSTM | Fuses both outputs → final refined forecast | Eq. 30–33 |

    ---

    ### 🌊 Hurst Exponent Decision Rule
    Before training, the **R/S analysis** (Hurst exponent) determines which model to use:
    - **H > 0.52** → Series has long-term memory → **Use full ARMA-RNN-LSTM hybrid**
    - **H ≈ 0.5**  → Random walk, no long-term memory → **Use RNN only** (hybrid adds noise)

    > *"For the SSE which lacks long-term memory, the RNN model achieves highest
    > forecasting accuracy"* — Xiao (2025) §4.3

    ---

    ### 📊 Data Sources
    | File | Contents |
    |------|----------|
    | `etf_price.parquet` | Daily closing prices (TLT, LQD, HYG, VNQ, GLD, SLV) |
    | `etf_ret.parquet` | Daily log-returns |
    | `etf_vol.parquet` | Daily volume |
    | `bench_price.parquet` | SPY + AGG benchmark prices |
    | `bench_ret.parquet` | SPY + AGG returns |

    ---

    ### ⚠️ Disclaimer
    This application is for **educational and research purposes only**.
    It does not constitute financial advice. Past model performance does not
    guarantee future results. Always consult a qualified financial advisor
    before making investment decisions.
    """)


# ── Live inference function ────────────────────────────────────────────────────
def _run_live_inference(price_df, ret_df, hurst_threshold, train_split):
    """Run the full pipeline in-browser with a progress bar."""
    if price_df is None:
        st.error("Cannot run inference — price data not loaded.")
        return

    from data_loader import build_feature_matrix, make_sequences, \
        train_test_split_sequences, Normaliser
    from hurst import hurst_exponent, classify_memory
    from trainer import train_pipeline
    import torch

    progress = st.progress(0, text="Starting inference…")
    results  = []
    device   = torch.device("cpu")   # Streamlit Cloud CPU only

    for i, etf in enumerate(TARGET_ETFS):
        progress.progress((i / len(TARGET_ETFS)), text=f"Processing {etf}…")

        if etf not in price_df.columns:
            continue

        data = {
            "price": price_df[[etf]].rename(columns={etf: etf}),
            "ret":   ret_df[[etf]] if ret_df is not None and etf in ret_df.columns
                     else pd.DataFrame(index=price_df.index),
            "vol":   pd.DataFrame(index=price_df.index),
            "bench_price": pd.DataFrame(index=price_df.index),
            "bench_ret":   pd.DataFrame(index=price_df.index),
            "bench_vol":   pd.DataFrame(index=price_df.index),
        }

        # Simplified data dict for feature builder
        data_full = {
            "price": price_df, "ret": ret_df or pd.DataFrame(),
            "vol": pd.DataFrame(index=price_df.index),
            "bench_price": pd.DataFrame(index=price_df.index),
            "bench_ret": pd.DataFrame(index=price_df.index),
            "bench_vol": pd.DataFrame(index=price_df.index),
        }

        ret_s = (ret_df[etf].dropna().values
                 if ret_df is not None and etf in ret_df.columns
                 else np.diff(np.log(price_df[etf].dropna().values)))
        H   = hurst_exponent(ret_s)
        mem = classify_memory(H)

        features = build_feature_matrix(data_full, etf)
        X, y, dates = make_sequences(features)
        if len(X) < 50:
            continue

        X_tr, y_tr, d_tr, X_te, y_te, d_te = \
            train_test_split_sequences(X, y, dates, train_split)
        norm = Normaliser()
        X_tr_n = norm.fit_transform(X_tr)
        X_te_n = norm.transform(X_te)

        res = train_pipeline(X_tr_n, y_tr, X_te_n, y_te, etf,
                             mem["use_hybrid"], device)

        current_price = float(price_df[etf].iloc[-1])
        pred_logret   = float(res["test_preds"][-1]) if len(res["test_preds"]) else 0.0
        pred_price    = current_price * np.exp(pred_logret)
        results.append({
            "etf": etf, "H": H, "model": mem["model"],
            "pred_ret_pct": pred_logret * 100,
            "pred_price": pred_price,
            "current_price": current_price,
            "dir_acc": res["metrics"].get("hybrid_dir_acc",
                       res["metrics"].get("rnn_dir_acc", 0)),
        })

    progress.progress(1.0, text="Done!")

    if results:
        results.sort(key=lambda x: x["pred_ret_pct"], reverse=True)
        st.success(f"✅ Inference complete! Top pick: **{results[0]['etf']}**")
        for r in results:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric(r["etf"], f"{r['pred_ret_pct']:+.3f}%")
            col_b.metric("Hurst H", f"{r['H']:.3f}")
            col_c.metric("Model", r["model"])
            col_d.metric("Dir Acc", f"{r['dir_acc']:.1f}%")
