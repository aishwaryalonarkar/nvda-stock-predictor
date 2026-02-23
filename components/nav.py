"""
components/nav.py
=================
Shared navigation bar injected into every page.
Single source of truth — change nav here, reflects everywhere.
"""

import streamlit as st


NAV_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Global resets ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Hide Streamlit default nav sidebar header ── */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Top navbar ── */
.nvda-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(10, 10, 20, 0.92);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 0 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    font-family: 'DM Sans', sans-serif;
    margin: -1rem -1rem 2rem -1rem; /* bleed to edges */
    width: calc(100% + 2rem);
}

.nvda-nav .brand {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    color: #ffffff;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.02em;
}

.nvda-nav .brand span {
    background: linear-gradient(135deg, #51CF66, #339AF0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.nvda-nav .links {
    display: flex;
    align-items: center;
    gap: 6px;
}

.nvda-nav .nav-link {
    color: rgba(255,255,255,0.6);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 8px;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    cursor: pointer;
    background: none;
    white-space: nowrap;
}

.nvda-nav .nav-link:hover {
    color: white;
    background: rgba(255,255,255,0.07);
}

.nvda-nav .nav-link.active {
    color: white;
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.12);
}

/* ── CTA Prediction button ── */
.nvda-nav .nav-cta {
    background: linear-gradient(135deg, #51CF66, #20C997);
    color: #0a0a14 !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 9px 20px !important;
    border-radius: 10px !important;
    border: none !important;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 18px rgba(81,207,102,0.3);
    transition: all 0.2s ease !important;
}

.nvda-nav .nav-cta:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(81,207,102,0.45) !important;
    background: linear-gradient(135deg, #69db7c, #38d9a9) !important;
}

/* ── Page base styles ── */
body, .stApp {
    background-color: #0a0a14;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1e !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #ccc !important; }
[data-testid="stSidebar"] label { color: #aaa !important; }
</style>
"""


def render_nav(active_page: str = "home"):
    """
    Inject the top navigation bar.

    Parameters
    ----------
    active_page : str
        "home" | "prediction"
        Highlights the matching nav link.
    """
    st.markdown(NAV_CSS, unsafe_allow_html=True)

    home_class = "nav-link active" if active_page == "home" else "nav-link"
    pred_class = "nav-link active" if active_page == "prediction" else "nav-link"

    st.markdown(f"""
    <nav class="nvda-nav">
        <a class="brand" href="/">📈 <span>NVDA</span> Predictor</a>
        <div class="links">
            <a class="{home_class}" href="/Home" target="_self">🏠 Home</a>
            <a class="{pred_class} nav-cta" href="/Prediction" target="_self">
                <strong>SHOW ME THE PREDICTION</strong>
            </a>
        </div>
    </nav>
    """, unsafe_allow_html=True)
