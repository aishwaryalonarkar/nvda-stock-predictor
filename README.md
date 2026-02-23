# NVDA Stock Price Predictor

## Quick Start

```bash
pip install -r requirements.txt
streamlit run main.py
```

Open http://localhost:8501

---

## Run with Docker (zero env issues)

```bash
docker-compose up --build
```

---

## Project Structure

```
.
├── main.py                    # Entry point — redirects to Home
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── pages/                     # Streamlit auto-discovers these as nav pages
│   ├── Home.py          # Homepage — friendly explainer (kids page)
│   └── Prediction.py    # Live XGBoost prediction dashboard
│
├── core/                      # Business logic — zero UI code here
│   ├── __init__.py
│   └── model.py               # Data fetching, feature engineering, training
│
└── components/                # Shared UI components
    ├── __init__.py
    └── nav.py                 # Top nav bar — single source of truth
```

### Design principles applied
- **Separation of concerns** — ML logic (core/) is completely separate from UI (pages/)
- **Single source of truth** — nav defined once in components/nav.py, imported everywhere
- **DRY** — no duplicated data fetching or feature engineering code
- **Streamlit native routing** — uses built-in multi-page system, no hacks
- **Caching at the data layer** — @st.cache_data lives in core/model.py, not in pages

---

## Navigation

| Page | Description |
|------|-------------|
| 🏠 Home | Kid-friendly explainer of how it works. Default homepage. |
| 📈 Prediction | Live model — configure ticker, run, explore charts |

Top nav bar on every page has a bold **SHOW ME THE PREDICTION** CTA.

---

## ⚠️ Disclaimer
Educational purposes only. Not financial advice.
