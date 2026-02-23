# 📈 NVDA Stock Price Predictor

A machine learning web app that predicts NVIDIA's stock price 5 days ahead — built entirely on free data sources, containerized with Docker, and served via a pre-trained model for instant predictions.

> **Disclaimer:** This is an educational project. Nothing here is financial advice.

---

## 🚀 Quick Start

```bash
git clone https://github.com/aishwaryalonarkar/nvda-stock-predictor.git
cd nvda-stock-predictor
docker-compose up --build
```

Open **http://localhost:8501** — no local Python, no pip installs, no environment issues.

On first visit to the Prediction page, click **Train Model Now** (runs once, ~2–3 min). Every prediction after that takes ~2 seconds.

---

## 📂 Project Structure

```
nvda-stock-predictor/
│
├── main.py                      # Entry point — redirects to Home page
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── pages/
│   ├── 1_🏠_Home.py            # Homepage — plain-English explainer of how it works
│   └── 2_📈_Prediction.py      # Prediction dashboard (UI only, no ML logic)
│
├── core/
│   ├── model.py                 # Data fetching, feature engineering, model training
│   ├── train.py                 # Offline training pipeline — run once, saves artifacts
│   └── serve.py                 # Serving layer — loads saved model, fetches 60d data, predicts
│
├── components/
│   └── nav.py                   # Shared top navigation bar (single source of truth)
│
└── models/                      # Trained model artifacts saved here (gitignored)
    ├── NVDA_5d_model.pkl
    ├── NVDA_5d_scaler.pkl
    ├── NVDA_5d_feature_cols.json
    ├── NVDA_5d_cv_metrics.csv
    └── NVDA_5d_meta.json
```

---

## 🗄️ Data Sources

All data is **free** — no Bloomberg Terminal, no paid API keys required.

| Source | Library | What we get | When fetched |
|--------|---------|-------------|--------------|
| **Yahoo Finance** | `yfinance` | 5 years of daily OHLCV (open, high, low, close, volume) | Training only |
| **Yahoo Finance** | `yfinance` | ~20 fundamental metrics: P/E, P/B, revenue growth, margins, ROE, analyst targets, short ratio, beta | Per prediction |
| **Yahoo Finance** | `yfinance` | Last 60 days of price data | Per prediction (fast) |
| **Yahoo Finance RSS** | `feedparser` | Recent NVDA news headlines | Per prediction |
| **Google News RSS** | `feedparser` | Broader NVIDIA / semiconductor headlines | Per prediction |
| **TextBlob NLP** | `textblob` | Sentiment score (−1 to +1) computed from each headline | Derived from RSS |

### Why not Bloomberg?

Bloomberg's full data API (BLPAPI) requires a Terminal subscription (~$25,000/year). Yahoo Finance via `yfinance` pulls from the same underlying SEC filings, exchange feeds, and analyst consensus data — it is the standard free alternative across quant research. The only meaningful gaps are real-time tick data and Bloomberg's proprietary alternative datasets (satellite imagery, credit card transaction data, etc.).

---

## 🧠 Model

### Algorithm: XGBoost (Extreme Gradient Boosting)

We use `XGBRegressor` to predict the **5-day forward percentage return** of NVDA.

**Why XGBoost and not something else?**

| Model | Verdict for this use case |
|-------|--------------------------|
| **XGBoost ✅** | Best fit for short-horizon tabular financial data. Fast to train, interpretable via feature importance, handles mixed feature types (price + fundamentals + sentiment) well, strong regularization prevents overfitting on noisy signals. |
| LSTM / Transformer | Better suited for very long sequences. Needs far more data and compute to outperform tree methods on a 5-day horizon with ~50 features. |
| Prophet | Designed for trend + seasonality decomposition. Not suited for high-frequency financial signals like RSI or news sentiment. |
| Linear Regression | Too simple — financial relationships are non-linear and feature interactions matter significantly. |

### Features (50+)

**Price-derived (technical indicators)**
- Returns over 1, 2, 3, 5, 10, 21 days
- Simple moving averages: SMA5, SMA10, SMA20, SMA50, SMA200
- Price vs SMA ratio (momentum signal)
- MACD, MACD signal line, MACD histogram
- RSI (14-day Relative Strength Index)
- Bollinger Bands: upper, lower, width, %B position
- ATR (Average True Range) — volatility proxy
- Volume ratio vs 20-day average
- Gap (open vs prior close), high-low daily range
- Rate of change: 5-day, 10-day
- Calendar features: day of week, month, quarter, month-end flag

**Fundamentals (from Yahoo Finance)**
- Trailing P/E, Forward P/E, Price-to-Book, Price-to-Sales
- Revenue growth YoY, Earnings growth YoY
- Gross margins, Operating margins
- Debt-to-equity, Current ratio
- Return on Equity (ROE), Return on Assets (ROA)
- Beta (market sensitivity)
- Short ratio (short interest signal)
- Analyst mean price target, low target, high target
- Number of analyst opinions
- Analyst recommendation mean (1 = Strong Buy, 5 = Sell)
- Earnings surprise % (most recent quarter)
- Analyst upside: (mean target / current price) − 1

**News Sentiment**
- Daily sentiment mean (average polarity across all headlines that day)
- Daily sentiment std (disagreement signal between headlines)
- News count (volume of coverage)
- 5-day rolling sentiment average
- 10-day rolling sentiment average

### Training Details

| Parameter | Value |
|-----------|-------|
| Training data | 5 years of daily NVDA history (~1,250 trading days) |
| Target variable | `close.pct_change(5).shift(-5)` — 5-day forward % return |
| Validation | `TimeSeriesSplit(n_splits=5)` walk-forward CV — no look-ahead bias |
| Scaler | `RobustScaler` — robust to earnings spike outliers |
| Trees | 500 estimators, max depth 4 |
| Regularization | L1 `reg_alpha=0.1` + L2 `reg_lambda=1.0` |
| Key metric | Directional Accuracy — did the model correctly predict up vs down? |

> 55–62% directional accuracy is considered meaningful in quantitative finance. A coin flip is 50%.

---

## ⚙️ System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                 (runs once / nightly)                       │
│                                                             │
│  Yahoo Finance ──► Feature Engineering ──► XGBoost Train   │
│  5yr OHLCV +       50+ indicators,             │           │
│  Fundamentals +    fundamentals,               ▼           │
│  News Sentiment    sentiment            models/*.pkl        │
└──────────────────────────────────────────────┬─────────────┘
                                               │
                                    Saved to disk
                                    (Docker volume mount —
                                     survives restarts)
                                               │
┌──────────────────────────────────────────────▼─────────────┐
│                    SERVING PIPELINE                         │
│               (every prediction request, ~2 sec)           │
│                                                             │
│  Load model.pkl ──► Fetch last 60d ──► Engineer ──► ⚡    │
│  (from disk,         of prices +       same               │
│   instant)           fresh news +      features           │
│                      fundamentals      → predict           │
└─────────────────────────────────────────────────────────────┘
```

Train once. Serve instantly. Training the model from scratch takes 2–3 minutes and only needs to happen when you want to update it with newer data. Every subsequent prediction loads the saved `.pkl` files and only fetches the last 60 days of market data, completing in around 2 seconds.

---

## ⚠️ Known Limitations

### Data
- `yfinance` has a ~15-minute delay on prices — not suitable for intraday or high-frequency trading strategies
- News sentiment uses rule-based TextBlob NLP, which misses sarcasm, context, and finance-specific language. A fine-tuned FinBERT model would score meaningfully better.
- RSS feeds only capture recent headlines (~2 weeks). Historical sentiment going back years would improve training quality.
- No options market data (implied volatility, put/call ratio) — these are strong short-term directional signals.
- No macro features: Fed funds rate, CPI prints, sector ETF flows (SOXX), VIX — all of which significantly affect NVDA's short-term price.

### Model
- XGBoost is a static model — it does not adapt to regime changes (e.g. 2022 rate hike environment vs 2024 AI boom). It needs periodic retraining to stay calibrated.
- The 5-day return target is inherently noisy. Stock prices contain a random component no model fully captures.
- Feature importance reflects correlation with past patterns, not causation. The model can be confidently wrong.
- No uncertainty quantification — output is a point prediction with no confidence interval or probability range.

### System
- Trained model artifacts are stored on local disk inside a Docker volume. In production these should live in object storage (AWS S3 / GCS) with versioning and rollback.
- No automated retraining schedule. Currently manual — should be a nightly cron job or an Airflow/Prefect DAG.
- Single-ticker design. Extending to arbitrary tickers requires pre-training or on-demand training per ticker with a model registry.
- No A/B testing framework to compare model versions before promoting to production.
- No monitoring or alerting for model drift or data pipeline failures.

---

## 🔮 Future Improvements

**Data**
- [ ] Replace TextBlob with **FinBERT** (finance-specific BERT model) for significantly better headline sentiment
- [ ] Add options data: implied volatility surface, put/call ratio from CBOE free feeds
- [ ] Add macro features: VIX, 10Y Treasury yield, SOXX ETF, DXY index
- [ ] Pull and score earnings call transcripts using NLP
- [ ] Historical news sentiment backfill via Tiingo News or Benzinga API

**Model**
- [ ] Add prediction confidence intervals using conformal prediction
- [ ] Ensemble: XGBoost + LightGBM + CatBoost majority vote
- [ ] SHAP values for per-prediction explainability ("why did it predict UP?")
- [ ] Automated nightly retraining with versioned model artifacts
- [ ] Backtesting engine: simulate live trades using past predictions and report Sharpe ratio, max drawdown, win rate

**System**
- [ ] Move model artifacts to S3/GCS with a versioned model registry (MLflow)
- [ ] GitHub Actions CI pipeline: lint → test → retrain → deploy on push to main
- [ ] Deploy to Railway / Render for a public URL directly from this repo
- [ ] Redis cache for serving predictions without redundant data fetches
- [ ] Multi-ticker support with a ticker selector and per-ticker model management
- [ ] Real-time price streaming via WebSocket (Alpaca or Polygon.io free tier)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Streamlit |
| ML model | XGBoost |
| Data | yfinance, feedparser, TextBlob |
| Visualization | Plotly |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11 |

---

## 🏃 Running Without Docker

```bash
pip install -r requirements.txt
streamlit run main.py
```

> **Mac users:** If you get a `libomp` error on XGBoost import, run `brew install libomp` first.

---

*Built by [@aishwaryalonarkar](https://github.com/aishwaryalonarkar)*