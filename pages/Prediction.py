"""
pages/2_📈_Prediction.py
=========================
Prediction dashboard — UI only.
All ML logic is in core/model.py + core/serve.py.

Flow:
  1. Check if trained model exists in models/
  2. If yes  → load artifacts + fetch 60d data → predict in ~2 sec
  3. If no   → show Train button → run offline training → save → predict
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.nav import render_nav
from core.serve import model_exists, predict
from core.train import train_and_save

st.set_page_config(
    page_title="NVDA Predictor — Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_nav(active_page="prediction")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
.stApp { background-color: #0a0a14; color: #e8e8f0; }
.main .block-container { max-width: 1200px; padding-top: 0 !important; }
h1,h2,h3 { color: #e8e8f0 !important; font-family: 'Syne', sans-serif !important; }
.metric-row { display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }
.mcard { background:#13132a; border-radius:16px; padding:20px 24px; border:1px solid rgba(255,255,255,0.07); flex:1; min-width:160px; text-align:center; transition:transform 0.2s; }
.mcard:hover { transform:translateY(-3px); }
.mcard .val { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; margin-bottom:4px; }
.mcard .lbl { font-size:0.75rem; color:#6666aa; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
.up   { color:#51CF66; }
.down { color:#FF6B6B; }
.badge-ok  { background:#0f2a0f; border:1px solid #51CF66; border-radius:10px; padding:8px 14px; font-size:0.82rem; color:#51CF66; font-weight:700; display:inline-block; margin-bottom:10px; }
.badge-warn{ background:#2a1a0f; border:1px solid #FF922B; border-radius:10px; padding:8px 14px; font-size:0.82rem; color:#FF922B; font-weight:700; display:inline-block; margin-bottom:10px; }
[data-testid="stSidebar"] { background:#0f0f1e !important; border-right:1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] * { color:#ccc; }
.stButton>button { background:linear-gradient(135deg,#51CF66,#20C997); color:#0a1f0a !important; border:none !important; border-radius:12px !important; padding:12px 0 !important; font-family:'Syne',sans-serif !important; font-weight:800 !important; font-size:0.95rem !important; width:100% !important; box-shadow:0 4px 18px rgba(81,207,102,0.25); }
.stTabs [data-baseweb="tab-list"] { background:#13132a; border-radius:12px; padding:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#aaa; font-weight:600; }
.stTabs [aria-selected="true"] { background:#1e1e3f; color:white !important; }
.disclaimer { background:#13132a; border-radius:12px; padding:14px 20px; border-left:4px solid #FF6B6B; font-size:0.82rem; color:#888; margin-top:28px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.page_link("pages/Home.py",   label="About")
    # Link to current page
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    ticker        = st.text_input("Ticker Symbol", value="NVDA").upper().strip()
    forecast_days = st.select_slider(
        "Forecast Horizon",
        options=[1, 3, 5, 10, 21], value=5,
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
    )
    train_years = st.slider("Training Years", 2, 10, 5,
                            help="Only used when training for the first time.")
    st.markdown("---")

    exists = model_exists(ticker, forecast_days)
    if exists:
        st.markdown('<div class="badge-ok">Model ready — instant prediction</div>', unsafe_allow_html=True)
        predict_btn = st.button("⚡ GET PREDICTION")
        retrain_btn = st.button("🔄 Retrain Model")
    else:
        st.markdown('<div class="badge-warn">⚠️ No model trained yet</div>', unsafe_allow_html=True)
        st.caption("Train once (~2–3 min). All future predictions take ~2 seconds.")
        predict_btn = False
        retrain_btn = st.button("🏋️ TRAIN MODEL NOW")

    st.markdown("---")
    st.markdown("""
**How it works**
- 🏋️ Train once → save to `models/`
- ⚡ Serve: load model + 60d data
- Prediction time: ~2 seconds
    """)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"# 📈 {ticker} Stock Price Predictor")
st.markdown("*XGBoost · Pre-trained model · Instant predictions*")

# ── Train ──────────────────────────────────────────────────────────────────
if retrain_btn:
    #     with st.status("🏋️ Training… (this runs once, then all predictions are instant)",
    #                    expanded=True) as status:
    #         st.write(f"📊 Fetching {train_years} years of {ticker} history…")
    #         st.write("🔧 Engineering 50+ features…")
    #         st.write("🌳 Training XGBoost with walk-forward CV…")
    #         try:
    #             meta = train_and_save(ticker, train_years, forecast_days)
    #             status.update(label="Model trained and saved!", state="complete")
    #             st.success(f"Done! Avg Directional Accuracy: {meta['avg_dir_acc']:.1f}%")
    #             st.rerun()
    #         except Exception as e:
    #             status.update(label="❌ Training failed", state="error")
    #             st.error(str(e))
    #     st.stop()
    # NEW FIXED CODE
    should_rerun = False

    with st.status("🏋️ Training...", expanded=True) as status:
        try:
            meta = train_and_save(ticker, train_years, forecast_days)
            status.update(label="Model trained and saved!", state="complete")
            should_rerun = True # Just set a flag here
        except Exception as e:
            status.update(label="❌ Training failed", state="error")
            st.error(str(e))

    if should_rerun:
        st.rerun() # Call it OUTSIDE the status block

if not predict_btn:
    if exists:
        st.info("👈 Hit **GET PREDICTION** in the sidebar.")
    else:
        st.info("👈 No model found for this ticker. Hit **TRAIN MODEL NOW** in the sidebar first.")
    st.stop()

# ── Predict ────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading saved model + fetching latest 60 days of data…"):
    try:
        result = predict(ticker, forecast_days)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

current_price    = result["current_price"]
predicted_price  = result["predicted_price"]
predicted_return = result["predicted_return"]
price_change     = result["price_change"]
price_df         = result["price_df"]
sentiment_df     = result["sentiment_df"]
raw_news_df      = result["raw_news_df"]
cv_df            = result["cv_df"]
model            = result["model"]
feature_cols     = result["feature_cols"]
is_up            = predicted_return > 0
dir_class        = "up" if is_up else "down"
dir_label        = "📈 UP" if is_up else "📉 DOWN"

st.markdown(
    f'<div class="badge-ok">✅ Model trained: {result["trained_at"][:10]} · '
    f'Dir. Accuracy: {result["avg_dir_acc"]:.1f}% · {result["n_features"]} features</div>',
    unsafe_allow_html=True,
)

st.markdown(f"""
<div class="metric-row">
  <div class="mcard"><div class="val">${current_price:.2f}</div><div class="lbl">Current Price</div></div>
  <div class="mcard"><div class="val {dir_class}">${predicted_price:.2f}</div><div class="lbl">Predicted in {forecast_days}d</div></div>
  <div class="mcard"><div class="val {dir_class}">{price_change:+.2f}</div><div class="lbl">$ Change</div></div>
  <div class="mcard"><div class="val {dir_class}">{predicted_return:+.2%}</div><div class="lbl">% Return</div></div>
  <div class="mcard"><div class="val {dir_class}">{dir_label}</div><div class="lbl">Signal</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart","🌳 Feature Importance","📋 CV Results","📰 News Sentiment"])

with tab1:
    recent = price_df.tail(90)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75,0.25], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"], name=ticker,
        increasing_line_color="#51CF66", decreasing_line_color="#FF6B6B",
    ), row=1, col=1)
    for w, c in [(20,"#339AF0"),(50,"#FFD43B")]:
        fig.add_trace(go.Scatter(x=recent.index, y=recent["close"].rolling(w).mean(),
                                  name=f"SMA{w}", line=dict(color=c,width=1.2,dash="dot")), row=1, col=1)
    fc_x = [price_df.index[-1], price_df.index[-1]+timedelta(days=forecast_days+2)]
    fig.add_trace(go.Scatter(x=fc_x, y=[current_price,predicted_price], mode="lines+markers",
        name=f"Forecast +{forecast_days}d",
        line=dict(color="#51CF66" if is_up else "#FF6B6B", width=2, dash="dash"),
        marker=dict(size=[8,14], symbol=["circle","star"])), row=1, col=1)
    vc = ["#51CF66" if c>=o else "#FF6B6B" for c,o in zip(recent["close"],recent["open"])]
    fig.add_trace(go.Bar(x=recent.index,y=recent["volume"],name="Volume",marker_color=vc,opacity=0.6), row=2, col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0a0a14",plot_bgcolor="#13132a",
        height=520,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importance.nlargest(20).sort_values()
    fig2 = px.bar(x=top20.values, y=top20.index, orientation="h", color=top20.values,
        color_continuous_scale=["#339AF0","#51CF66","#FFD43B"],
        title="Top 20 Feature Importances", labels={"x":"Importance Score","y":""})
    fig2.update_layout(template="plotly_dark",paper_bgcolor="#0a0a14",plot_bgcolor="#13132a",
        height=520,showlegend=False,coloraxis_showscale=False,margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if not cv_df.empty:
        st.markdown("**Walk-Forward CV** — trained on past, tested on future. No look-ahead bias.")
        st.dataframe(cv_df.style.background_gradient(subset=["MAE","RMSE"],cmap="RdYlGn_r"),
                     use_container_width=True, hide_index=True)
        avg = cv_df["Dir. Accuracy"].str.rstrip("%").astype(float).mean()
        st.metric("Average Directional Accuracy", f"{avg:.1f}%", help=">55% is meaningful in finance.")
    else:
        st.info("CV metrics not available — retrain to generate.")

with tab4:
    if not sentiment_df.empty:
        fig4 = px.bar(sentiment_df.tail(30), x="date", y="sentiment_mean",
            color="sentiment_mean", color_continuous_scale=["#FF6B6B","#555577","#51CF66"],
            range_color=[-0.5,0.5], title="Daily News Sentiment (last 30 days)",
            labels={"sentiment_mean":"Sentiment","date":""})
        fig4.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.25)
        fig4.update_layout(template="plotly_dark",paper_bgcolor="#0a0a14",plot_bgcolor="#13132a",
            height=300,coloraxis_showscale=False,margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig4, use_container_width=True)
        if not raw_news_df.empty:
            st.markdown("**Recent Headlines**")
            d = raw_news_df[["date","headline","polarity"]].sort_values("date",ascending=False).head(20).copy()
            d["polarity"] = d["polarity"].round(3)
            d["sentiment"] = d["polarity"].apply(lambda x:"🟢 Positive" if x>0.05 else("🔴 Negative" if x<-0.05 else "⚪ Neutral"))
            st.dataframe(d[["date","headline","sentiment","polarity"]], use_container_width=True, hide_index=True)
    else:
        st.warning("No news sentiment data fetched.")

st.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> Educational and research purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)