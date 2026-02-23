"""
core/model.py
=============
All data fetching, feature engineering, and model training logic.
Imported by pages — never contains UI code.
This is the single source of truth for the ML pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st

try:
    import yfinance as yf
    from xgboost import XGBRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from textblob import TextBlob
    import feedparser
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    MISSING_DEP = str(e)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────


# def fetch_price_data(ticker, years):
#     import yfinance as yf
#     data = yf.download(ticker, period=f"{years}y")
    
#     # --- ADD THESE TWO LINES TO FIX THE TUPLE ERROR ---
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)
#     # --------------------------------------------------
    
#     # Now your existing code (which likely calls .lower()) will work:
#     data.columns = [str(x).lower() for x in data.columns] 
#     return data

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, years: int) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=years * 365)
    df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    fundamentals = {
        "pe_ratio":             info.get("trailingPE",                       np.nan),
        "forward_pe":           info.get("forwardPE",                        np.nan),
        "pb_ratio":             info.get("priceToBook",                      np.nan),
        "ps_ratio":             info.get("priceToSalesTrailing12Months",     np.nan),
        "revenue_growth":       info.get("revenueGrowth",                    np.nan),
        "earnings_growth":      info.get("earningsGrowth",                   np.nan),
        "gross_margins":        info.get("grossMargins",                     np.nan),
        "operating_margins":    info.get("operatingMargins",                 np.nan),
        "debt_to_equity":       info.get("debtToEquity",                     np.nan),
        "current_ratio":        info.get("currentRatio",                     np.nan),
        "return_on_equity":     info.get("returnOnEquity",                   np.nan),
        "return_on_assets":     info.get("returnOnAssets",                   np.nan),
        "beta":                 info.get("beta",                             np.nan),
        "short_ratio":          info.get("shortRatio",                       np.nan),
        "analyst_target":       info.get("targetMeanPrice",                  np.nan),
        "analyst_low":          info.get("targetLowPrice",                   np.nan),
        "analyst_high":         info.get("targetHighPrice",                  np.nan),
        "num_analyst_opinions": info.get("numberOfAnalystOpinions",          np.nan),
        "recommendation_mean":  info.get("recommendationMean",               np.nan),
    }
    try:
        t  = yf.Ticker(ticker)
        eh = t.earnings_history
        fundamentals["earnings_surprise_pct"] = (
            float(eh.iloc[-1].get("surprisePercent", np.nan))
            if eh is not None and not eh.empty else np.nan
        )
    except Exception:
        fundamentals["earnings_surprise_pct"] = np.nan
    return fundamentals


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_sentiment(ticker: str, company_name: str = "NVIDIA"):
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={company_name}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={ticker}+NVIDIA+semiconductor&hl=en-US&gl=US&ceid=US:en",
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
                        "date":         datetime(*pub[:3]),
                        "polarity":     blob.sentiment.polarity,
                        "subjectivity": blob.sentiment.subjectivity,
                        "headline":     entry.get("title", ""),
                    })
        except Exception:
            pass

    empty = pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_std", "news_count"])
    if not records:
        return empty, pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby("date").agg(
        sentiment_mean=("polarity", "mean"),
        sentiment_std =("polarity", "std"),
        news_count    =("polarity", "count"),
    ).reset_index()
    return daily, df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["close"], df["volume"]

    for d in [1, 2, 3, 5, 10, 21]:
        df[f"return_{d}d"] = c.pct_change(d)

    for w in [5, 10, 20, 50, 200]:
        df[f"sma_{w}"]          = c.rolling(w).mean()
        df[f"price_vs_sma{w}"]  = (c / df[f"sma_{w}"]) - 1

    df["ema_12"]      = c.ewm(span=12).mean()
    df["ema_26"]      = c.ewm(span=26).mean()
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    delta       = c.diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    hi, lo = df["high"], df["low"]
    tr           = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"]= df["atr_14"] / c

    df["vol_sma20"]    = v.rolling(20).mean()
    df["vol_ratio"]    = v / df["vol_sma20"]
    df["vol_change"]   = v.pct_change()
    df["gap"]          = (df["open"] - c.shift()) / c.shift()
    df["high_low_pct"] = (hi - lo) / c
    df["close_vs_high"]= (c - hi) / (hi - lo + 1e-9)
    df["roc_5"]        = c.pct_change(5)
    df["roc_10"]       = c.pct_change(10)
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["is_month_end"] = df.index.is_month_end.astype(int)
    return df


def build_features(price_df, fundamentals, sentiment_df, forecast_days=5):
    df = price_df.copy()
    df = add_technical_indicators(df)
    df["target"] = df["close"].pct_change(forecast_days).shift(-forecast_days)

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
    df = df.dropna(subset=["target"])

    exclude      = {"open","high","low","close","volume","target"}
    feature_cols = [c for c in df.columns if c not in exclude]
    nan_pct      = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if nan_pct[c] < 0.5]
    X = df[feature_cols].ffill().fillna(0)
    y = df["target"]
    return X, y, feature_cols


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def run_training(X: pd.DataFrame, y: pd.Series):
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        random_state=42, early_stopping_rounds=30,
        eval_metric="mae", verbosity=0,
    )

    tscv       = TimeSeriesSplit(n_splits=5)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y.iloc[train_idx],   y.iloc[val_idx]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds   = model.predict(X_val)
        mae     = mean_absolute_error(y_val, preds)
        rmse    = np.sqrt(mean_squared_error(y_val, preds))
        r2      = r2_score(y_val, preds)
        dir_acc = np.mean(np.sign(preds) == np.sign(y_val))
        cv_metrics.append({
            "Fold": fold + 1,
            "MAE":  round(mae,  4),
            "RMSE": round(rmse, 4),
            "R²":   round(r2,   3),
            "Dir. Accuracy": f"{dir_acc:.1%}",
        })

    model.fit(X_scaled, y, eval_set=[(X_scaled, y)], verbose=False)
    return model, scaler, pd.DataFrame(cv_metrics)