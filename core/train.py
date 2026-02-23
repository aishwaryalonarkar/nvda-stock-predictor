"""
core/train.py
=============
OFFLINE TRAINING PIPELINE — run once (or nightly via cron/scheduler).
Produces model artifacts saved to models/ directory.
The serving layer (Streamlit) loads these artifacts — never retrains live.

Usage:
    python -m core.train
    python -m core.train --ticker NVDA --years 5
"""

import argparse
import os
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd

from core.model import (
    fetch_price_data,
    fetch_fundamentals,
    fetch_news_sentiment,
    build_features,
    run_training,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_and_save(ticker: str = "NVDA", years: int = 5, forecast_days: int = 5):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  OFFLINE TRAINING PIPELINE")
    print(f"  Ticker: {ticker} | Years: {years} | Horizon: {forecast_days}d")
    print(f"{'='*55}\n")

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    print("[1/5] Fetching price data...")
    price_df = fetch_price_data(ticker, years)
    print(f"      {len(price_df)} trading days loaded.")

    print("[2/5] Fetching fundamentals...")
    fundamentals = fetch_fundamentals(ticker)

    print("[3/5] Fetching news sentiment...")
    result = fetch_news_sentiment(ticker)
    sentiment_df = result[0] if isinstance(result, tuple) else result

    # ── 2. Feature engineering ────────────────────────────────────────────
    print("[4/5] Engineering features...")
    X, y, feature_cols = build_features(price_df, fundamentals, sentiment_df, forecast_days)

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("[5/5] Training XGBoost...")
    model, scaler, cv_df = run_training(X, y)

    # ── 4. Save artifacts ─────────────────────────────────────────────────
    slug = f"{ticker}_{forecast_days}d"

    # Model + scaler
    with open(os.path.join(MODELS_DIR, f"{slug}_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, f"{slug}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Feature column names (needed to align live data at serve time)
    with open(os.path.join(MODELS_DIR, f"{slug}_feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    # CV metrics for display
    cv_df.to_csv(os.path.join(MODELS_DIR, f"{slug}_cv_metrics.csv"), index=False)

    # Training metadata
    meta = {
        "ticker":        ticker,
        "forecast_days": forecast_days,
        "train_years":   years,
        "trained_at":    datetime.utcnow().isoformat() + "Z",
        "n_samples":     len(X),
        "n_features":    len(feature_cols),
        "avg_dir_acc":   cv_df["Dir. Accuracy"].str.rstrip("%").astype(float).mean(),
        "feature_importance": dict(zip(feature_cols,
                                       model.feature_importances_.tolist())),
    }
    with open(os.path.join(MODELS_DIR, f"{slug}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Artifacts saved to models/{slug}_*")
    print(f"   Avg Directional Accuracy: {meta['avg_dir_acc']:.1f}%")
    print(f"   Features: {len(feature_cols)}, Samples: {len(X)}")
    print(f"   Trained at: {meta['trained_at']}\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NVDA XGBoost model offline.")
    parser.add_argument("--ticker",   default="NVDA", help="Stock ticker symbol")
    parser.add_argument("--years",    default=5, type=int, help="Years of training data")
    parser.add_argument("--horizon",  default=5, type=int, help="Forecast days ahead")
    args = parser.parse_args()
    train_and_save(args.ticker, args.years, args.horizon)