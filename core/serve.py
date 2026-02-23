"""
core/serve.py
=============
ONLINE SERVING LAYER — called on every prediction request.
Loads pre-trained model artifacts from models/ directory.
Only fetches the last 60 days of price data (fast).
Total prediction time: ~2-3 seconds vs ~3-5 minutes for full retrain.

Architecture:
    Training (offline, once)  →  models/*.pkl saved to disk
    Serving  (online, fast)   →  load pkl + fetch 60d data + predict
"""

import os
import pickle
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def model_exists(ticker: str, forecast_days: int) -> bool:
    """Check if trained artifacts exist for this ticker/horizon."""
    slug = f"{ticker}_{forecast_days}d"
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f"{slug}_{ext}"))
        for ext in ["model.pkl", "scaler.pkl", "feature_cols.json", "meta.json"]
    )


def load_artifacts(ticker: str, forecast_days: int) -> dict:
    """Load saved model, scaler, feature list, and metadata."""
    slug = f"{ticker}_{forecast_days}d"

    with open(os.path.join(MODELS_DIR, f"{slug}_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, f"{slug}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, f"{slug}_feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODELS_DIR, f"{slug}_meta.json")) as f:
        meta = json.load(f)

    cv_path = os.path.join(MODELS_DIR, f"{slug}_cv_metrics.csv")
    cv_df   = pd.read_csv(cv_path) if os.path.exists(cv_path) else pd.DataFrame()

    return {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "meta":         meta,
        "cv_df":        cv_df,
    }


@st.cache_data(ttl=900, show_spinner=False)   # Cache 15 min — fresh enough for intraday
def fetch_recent_data(ticker: str, days: int = 60):
    """
    Fetch only the last N days of price + recent fundamentals + recent news.
    Much faster than fetching 5 years every time.
    """
    import yfinance as yf
    from textblob import TextBlob
    import feedparser

    # Price — only last 60 days (enough for all rolling indicators up to SMA50)
    end   = datetime.today()
    start = end - timedelta(days=days)
    price_df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    price_df.columns = [c.lower() for c in price_df.columns]
    price_df.index.name = "date"

    # Fundamentals — scalar, fast
    info = yf.Ticker(ticker).info
    fundamentals = {
        "pe_ratio":             info.get("trailingPE",                   np.nan),
        "forward_pe":           info.get("forwardPE",                    np.nan),
        "pb_ratio":             info.get("priceToBook",                  np.nan),
        "ps_ratio":             info.get("priceToSalesTrailing12Months", np.nan),
        "revenue_growth":       info.get("revenueGrowth",                np.nan),
        "earnings_growth":      info.get("earningsGrowth",               np.nan),
        "gross_margins":        info.get("grossMargins",                 np.nan),
        "operating_margins":    info.get("operatingMargins",             np.nan),
        "debt_to_equity":       info.get("debtToEquity",                 np.nan),
        "current_ratio":        info.get("currentRatio",                 np.nan),
        "return_on_equity":     info.get("returnOnEquity",               np.nan),
        "return_on_assets":     info.get("returnOnAssets",               np.nan),
        "beta":                 info.get("beta",                         np.nan),
        "short_ratio":          info.get("shortRatio",                   np.nan),
        "analyst_target":       info.get("targetMeanPrice",              np.nan),
        "analyst_low":          info.get("targetLowPrice",               np.nan),
        "analyst_high":         info.get("targetHighPrice",              np.nan),
        "num_analyst_opinions": info.get("numberOfAnalystOpinions",      np.nan),
        "recommendation_mean":  info.get("recommendationMean",           np.nan),
        "earnings_surprise_pct": np.nan,
    }

    # News sentiment — recent only
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q=NVIDIA+stock&hl=en-US&gl=US&ceid=US:en",
    ]
    records = []
    for url in feeds:
        try:
            for entry in feedparser.parse(url).entries:
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    text = entry.get("title", "") + " " + entry.get("summary", "")
                    blob = TextBlob(text)
                    records.append({
                        "date":     datetime(*pub[:3]),
                        "polarity": blob.sentiment.polarity,
                        "headline": entry.get("title", ""),
                    })
        except Exception:
            pass

    sentiment_df = pd.DataFrame()
    raw_news_df  = pd.DataFrame()
    if records:
        raw_news_df = pd.DataFrame(records)
        raw_news_df["date"] = pd.to_datetime(raw_news_df["date"]).dt.normalize()
        sentiment_df = raw_news_df.groupby("date").agg(
            sentiment_mean=("polarity", "mean"),
            sentiment_std =("polarity", "std"),
            news_count    =("polarity", "count"),
        ).reset_index()

    return price_df, fundamentals, sentiment_df, raw_news_df


def predict(ticker: str, forecast_days: int) -> dict:
    """
    Main serving function.
    Loads model → fetches recent data → engineers features → returns prediction.
    """
    from core.model import add_technical_indicators

    artifacts    = load_artifacts(ticker, forecast_days)
    model        = artifacts["model"]
    scaler       = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]
    meta         = artifacts["meta"]

    # Need enough history for SMA200 (largest rolling window) → fetch 300 days
    price_df, fundamentals, sentiment_df, raw_news_df = fetch_recent_data(ticker, days=300)

    if price_df.empty:
        raise ValueError(f"Could not fetch price data for {ticker}")

    # ── Engineer features (same pipeline as training) ──────────────────────
    df = price_df.copy()
    df = add_technical_indicators(df)

    for k, v in fundamentals.items():
        df[k] = v
    df["analyst_upside"] = (df["analyst_target"] / df["close"]) - 1

    if not sentiment_df.empty:
        s = sentiment_df.set_index("date")
        s.index = pd.to_datetime(s.index)
        df = df.join(s, how="left")
        for col in ["sentiment_mean", "sentiment_std", "news_count"]:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)
    else:
        df["sentiment_mean"] = 0.0
        df["sentiment_std"]  = 0.0
        df["news_count"]     = 0.0

    df["sentiment_5d_avg"]  = df["sentiment_mean"].rolling(5).mean()
    df["sentiment_10d_avg"] = df["sentiment_mean"].rolling(10).mean()
    df = df.ffill().fillna(0)

    # ── Align features to training columns exactly ─────────────────────────
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0                    # fill any new/missing features with 0

    latest         = df[feature_cols].iloc[[-1]]
    latest_scaled  = scaler.transform(latest)
    pred_return    = float(model.predict(latest_scaled)[0])

    current_price  = float(price_df["close"].iloc[-1])
    pred_price     = current_price * (1 + pred_return)

    return {
        "ticker":           ticker,
        "current_price":    current_price,
        "predicted_price":  pred_price,
        "predicted_return": pred_return,
        "price_change":     pred_price - current_price,
        "forecast_days":    forecast_days,
        "as_of_date":       price_df.index[-1].strftime("%Y-%m-%d"),
        "trained_at":       meta.get("trained_at", "unknown"),
        "avg_dir_acc":      meta.get("avg_dir_acc", 0),
        "n_features":       meta.get("n_features", 0),
        "price_df":         price_df,
        "sentiment_df":     sentiment_df,
        "raw_news_df":      raw_news_df,
        "cv_df":            artifacts["cv_df"],
        "model":            model,
        "feature_cols":     feature_cols,
    }