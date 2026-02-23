"""
pages/1_🏠_Home.py
==================
Homepage — friendly explainer. Uses st.components.v1.html()
to render rich HTML safely inside Streamlit.
"""

import streamlit as st
import streamlit.components.v1 as components
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.nav import render_nav

st.set_page_config(
    page_title="NVDA Predictor — Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_nav(active_page="home")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Pages")
    st.page_link("pages/Home.py",       label="🏠 Home  *(you are here)*")
    st.page_link("pages/Prediction.py", label="📈 Prediction")
    st.markdown("---")
    st.markdown("*NVDA Stock Predictor*")

# ── Page title ─────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: #ddf0fd; }
    .main .block-container { max-width: 880px; padding-top: 0 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
## 📈 NVDA Stock Price Prediction
##### Using free data from Yahoo Finance, Google News & TextBlob sentiment — powered by XGBoost
""")
st.markdown("---")

# ── Render the kids explainer as a self-contained HTML component ────────────
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:transparent;font-family:'Nunito',sans-serif;color:#1a1a2e;padding:8px 4px 40px;}
.hero{text-align:center;padding:32px 20px 24px;}
.hero-icon{font-size:72px;display:block;animation:bounce 2.5s ease-in-out infinite;filter:drop-shadow(0 8px 16px rgba(0,0,0,0.1));}
.hero h1{font-family:'Fredoka One',cursive;font-size:clamp(1.7rem,5vw,2.8rem);color:#1a1a2e;margin-top:12px;line-height:1.1;}
.hero p{font-size:1.05rem;color:#4a4a7a;margin-top:10px;font-weight:700;max-width:520px;margin-inline:auto;}
.card{background:white;border-radius:24px;padding:28px 32px;margin-top:20px;box-shadow:0 6px 28px rgba(0,0,0,0.07);}
.card-header{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.card-icon{font-size:1.9rem;background:#e8f4fd;width:56px;height:56px;border-radius:16px;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 3px 8px rgba(0,0,0,0.07);}
.card-title{font-family:'Fredoka One',cursive;font-size:1.45rem;color:#1a1a2e;}
.card-sub{font-size:0.82rem;color:#8888aa;font-weight:700;margin-top:1px;}
.card-body{font-size:0.98rem;line-height:1.75;color:#333;font-weight:600;}
.analogy{background:linear-gradient(135deg,#FFF3CD,#FFEAA7);border-radius:14px;padding:16px 20px;margin-top:14px;border-left:5px solid #FFD43B;font-size:0.94rem;font-weight:700;color:#7d5a00;}
.analogy::before{content:"🌟 Think of it like this: ";font-weight:800;}
.clue-item{background:#e8f4fd;border-radius:12px;padding:12px 16px;display:flex;align-items:flex-start;gap:10px;margin-bottom:9px;border-left:4px solid #339AF0;}
.clue-emoji{font-size:1.4rem;flex-shrink:0;margin-top:2px;}
.clue-q{font-weight:800;font-size:0.93rem;color:#1a1a2e;margin-bottom:3px;}
.clue-a{font-size:0.84rem;color:#20C997;font-weight:700;line-height:1.5;}
.tree-demo{background:linear-gradient(135deg,#f0fff4,#e0f7fa);border-radius:16px;padding:20px;margin-top:14px;text-align:center;}
.tree-demo h3{font-family:'Fredoka One',cursive;font-size:1.05rem;margin-bottom:14px;color:#1a1a2e;}
.tree-row{display:flex;justify-content:center;gap:10px;margin:5px 0;flex-wrap:wrap;}
.tnode{background:white;border-radius:10px;padding:8px 14px;box-shadow:0 3px 8px rgba(0,0,0,0.08);font-size:0.8rem;font-weight:700;border:2px solid #ddd;}
.tnode.root{background:#1a1a2e;color:white;border-color:#1a1a2e;}
.tnode.yes{background:#e8ffe8;border-color:#51CF66;color:#1e6b1e;}
.tnode.no{background:#ffe8e8;border-color:#FF6B6B;color:#6b1e1e;}
.tnode.mid{background:#fff8e0;border-color:#FFD43B;color:#7d5a00;}
.step{display:flex;gap:12px;align-items:flex-start;margin-bottom:16px;}
.step-num{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Fredoka One',cursive;font-size:1.1rem;color:white;flex-shrink:0;box-shadow:0 3px 10px rgba(0,0,0,0.15);}
.step-title{font-family:'Fredoka One',cursive;font-size:1.05rem;margin-bottom:2px;color:#1a1a2e;}
.step-desc{font-size:0.9rem;color:#555;font-weight:600;line-height:1.6;}
.result-card{background:linear-gradient(135deg,#1a1a2e,#2d2d54);color:white;border-radius:24px;padding:28px;margin-top:20px;text-align:center;box-shadow:0 12px 40px rgba(0,0,0,0.18);}
.result-card h2{font-family:'Fredoka One',cursive;font-size:1.6rem;margin-bottom:8px;}
.result-big{font-size:2.6rem;display:block;margin:12px 0 8px;}
.result-card p{font-size:0.95rem;opacity:0.82;font-weight:600;line-height:1.7;max-width:480px;margin-inline:auto;}
.chips{display:flex;justify-content:center;flex-wrap:wrap;gap:7px;margin-top:16px;}
.chip{background:rgba(255,255,255,0.1);border-radius:100px;padding:6px 14px;font-size:0.8rem;font-weight:800;border:1px solid rgba(255,255,255,0.15);}
.chip.g{background:rgba(81,207,102,0.22);border-color:rgba(81,207,102,0.4);}
.chip.y{background:rgba(255,212,59,0.22);border-color:rgba(255,212,59,0.4);}
.disclaimer{text-align:center;font-size:0.78rem;color:#aaa;font-weight:700;margin-top:20px;padding:12px;background:white;border-radius:10px;}
@keyframes bounce{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}
</style>
</head>
<body>

<div class="hero">
  <span class="hero-icon">🤖</span>
  <h1>How does our Stock Robot think?</h1>
  <p>We taught a computer to guess if NVIDIA's stock will go up or down — here's how!</p>
</div>

<!-- CARD 1: What is a stock -->
<div class="card">
  <div class="card-header">
    <div class="card-icon">🍫</div>
    <div>
      <div class="card-title">First — what IS a stock?</div>
      <div class="card-sub">Let's start from the very beginning</div>
    </div>
  </div>
  <div class="card-body">
    Imagine your friend has a lemonade stand 🍋. It's doing really well! She says:
    <b>"Give me $1 and I'll give you a tiny piece of my business."</b>
    That little piece is called a <b>stock</b>.<br><br>
    <b>NVIDIA</b> makes the special computer chips that power video games and AI robots.
    Millions of people own tiny pieces of NVIDIA — and every day, those pieces go up or
    down in price depending on how well the company is doing.
  </div>
  <div class="analogy">
    If NVIDIA has an amazing day and sells tons of chips, your piece becomes worth MORE.
    If something goes wrong, it's worth LESS. That price change — every single day —
    is what we're trying to predict!
  </div>
</div>

<!-- CARD 2: Clues -->
<div class="card">
  <div class="card-header">
    <div class="card-icon">🔍</div>
    <div>
      <div class="card-title">What clues does the robot look at?</div>
      <div class="card-sub">Just like a detective collecting evidence!</div>
    </div>
  </div>
  <div class="card-body">Our robot collects <b>clues</b> every single day before making its guess:</div>
  <div style="margin-top:14px;">
    <div class="clue-item"><div class="clue-emoji">📈</div><div><div class="clue-q">Past prices & how fast they changed</div><div class="clue-a">If the price has been climbing for 5 days, it might keep climbing! The robot watches speed and direction of price moves.</div></div></div>
    <div class="clue-item"><div class="clue-emoji">🎢</div><div><div class="clue-q">How bouncy has the price been? (RSI & Bollinger Bands)</div><div class="clue-a">Like a "bounce meter." If the price shot up too fast, it might bounce back down — like a ball thrown too high!</div></div></div>
    <div class="clue-item"><div class="clue-emoji">📰</div><div><div class="clue-q">What are people saying in the news?</div><div class="clue-a">The robot reads thousands of headlines and scores them happy 😊 or sad 😢. Lots of happy news usually means the price goes up!</div></div></div>
    <div class="clue-item"><div class="clue-emoji">💰</div><div><div class="clue-q">Is the company making money? (Fundamentals)</div><div class="clue-a">Is NVIDIA earning a LOT per chip? Are sales growing? Healthy companies usually have prices that rise over time!</div></div></div>
    <div class="clue-item"><div class="clue-emoji">👩‍💼</div><div><div class="clue-q">What do the experts think? (Analyst targets)</div><div class="clue-a">Smart people called "analysts" study NVIDIA all day. If their target is WAY higher than today's price — that's a big clue!</div></div></div>
    <div class="clue-item"><div class="clue-emoji">📦</div><div><div class="clue-q">How much are people trading? (Volume)</div><div class="clue-a">If TONS of people suddenly buy NVIDIA, that's exciting — like everyone rushing to a new ice cream flavor!</div></div></div>
  </div>
</div>

<!-- CARD 3: XGBoost -->
<div class="card">
  <div class="card-header">
    <div class="card-icon">🌳</div>
    <div>
      <div class="card-title">The robot's brain: XGBoost</div>
      <div class="card-sub">500 little detectives working as a team!</div>
    </div>
  </div>
  <div class="card-body">
    Our robot is called <b>XGBoost</b>. It's not ONE smart robot — it's <b>500 tiny robots</b>,
    each one asking simple Yes/No questions. They all vote — and the majority wins!
  </div>
  <div class="tree-demo">
    <h3>🌳 One tiny decision tree (simplified!)</h3>
    <div class="tree-row"><div class="tnode root">Is the price higher than 5 days ago?</div></div>
    <div class="tree-row" style="color:#aaa;font-size:0.75rem;font-weight:700;">
      <span style="margin-right:50px;">YES</span><span>NO ❌</span>
    </div>
    <div class="tree-row">
      <div class="tnode mid">Is the news happy?</div>
      <div class="tnode mid">Is the company making money?</div>
    </div>
    <div class="tree-row">
      <div class="tnode yes">📈 Probably going UP!</div>
      <div class="tnode no">📉 Might go DOWN.</div>
    </div>
    <p style="margin-top:14px;font-size:0.82rem;color:#777;font-weight:700;">
      One tree ≈ 4 questions. 500 trees × 4 questions = 2,000 questions at once! 🤯
    </p>
  </div>
  <div class="analogy">
    Imagine asking 500 friends "will it rain tomorrow?" and going with whatever MOST say.
    That's way smarter than asking just one friend! XGBoost does the same — for stock prices.
  </div>
</div>

<!-- CARD 4: Training -->
<div class="card">
  <div class="card-header">
    <div class="card-icon">🏋️</div>
    <div>
      <div class="card-title">How the robot "practiced"</div>
      <div class="card-sub">5 years of homework!</div>
    </div>
  </div>
  <div class="card-body">Before making guesses, the robot practiced on <b>5 years of NVIDIA history</b> — over 1,200 trading days. And here's the clever part: the model is <b>trained once</b> and saved — so predictions happen in seconds!</div>
  <div style="margin-top:16px;">
    <div class="step"><div class="step-num" style="background:#339AF0;">1</div><div><div class="step-title">📚 Look at old data</div><div class="step-desc">"On this day in 2020, the news was happy, the price had been rising, and 5 days later it went UP 8%!"</div></div></div>
    <div class="step"><div class="step-num" style="background:#FF922B;">2</div><div><div class="step-title">🤔 Make a guess</div><div class="step-desc">The robot tries to guess what happened next. At first, its guesses are terrible — it's still learning!</div></div></div>
    <div class="step"><div class="step-num" style="background:#FF6B6B;">3</div><div><div class="step-title">❌ Check the mistake</div><div class="step-desc">"You guessed +3%, the real answer was +8%." Like a homework grade!</div></div></div>
    <div class="step"><div class="step-num" style="background:#51CF66;">4</div><div><div class="step-title">✅ Save the trained brain</div><div class="step-desc">Once trained, the model is saved to disk. The next prediction just loads it — no waiting!</div></div></div>
    <div class="step"><div class="step-num" style="background:#CC5DE8;">5</div><div><div class="step-title">⚡ Serve instantly</div><div class="step-desc">When you click "Run Prediction", it only fetches the last 60 days of data and runs through the saved model in ~2 seconds.</div></div></div>
  </div>
</div>

<!-- Result card -->
<div class="result-card">
  <h2>🤖 So what does the robot actually do?</h2>
  <span class="result-big">👀 → 🧠 → ⚡ → 📊</span>
  <p>
    <b>Train once</b> on 5 years of data. <b>Save</b> the brain. Then every request:
    fetch only the last 60 days, run through the saved model, get a prediction in seconds!
  </p>
  <div class="chips">
    <div class="chip y">📊 50+ clues every day</div>
    <div class="chip g">🌳 500 decision trees</div>
    <div class="chip">📅 Trained on 5 years</div>
    <div class="chip g">⚡ Predicts in ~2 seconds</div>
  </div>
</div>

<div class="disclaimer">
  ⚠️ Remember: Even the smartest robot can't predict the future perfectly.
  Stocks are unpredictable — never invest money you can't afford to lose! 😄
</div>

</body>
</html>
"""

# components.html(HOME_HTML, height=6000, scrolling=False)
# Replace the components.html line with this:
st.components.v1.html(
    f"""
    <div style="height: 100%; overflow-y: auto;">
        {HOME_HTML}
    </div>
    """,
    height=3200 if st.sidebar.expanded else 5000,
    scrolling=True
)