

import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Page configuration

st.set_page_config(
    page_title="Cardio Monitor · Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("model")


def html_block(markup: str) -> None:
    """Render HTML without Markdown turning indented lines into code blocks."""
    clean = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)



# Load the EXISTING trained model and preprocessing artifacts.

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "LR_model_heart.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    expected_columns = joblib.load(MODEL_DIR / "columns.pkl")
    return model, scaler, list(expected_columns)


try:
    model, scaler, expected_columns = load_artifacts()
    artifacts_loaded = True
    load_error = None
except Exception as exc:  # noqa: BLE001
    model = scaler = expected_columns = None
    artifacts_loaded = False
    load_error = str(exc)


# Styling — dark "cardio monitor" theme

html_block(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
    :root {
        --bg: #070B0A;
        --panel: #0E1513;
        --panel-2: #121C19;
        --line: #1F2E2A;
        --text: #E6F2EE;
        --muted: #7F9A92;
        --pulse: #2BF5A0;
        --pulse-dim: rgba(43, 245, 160, 0.14);
        --alarm: #FF4D5E;
        --alarm-dim: rgba(255, 77, 94, 0.14);
        --amber: #FFC24B;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background:
            radial-gradient(1000px 500px at 100% 0%, rgba(43,245,160,0.06), transparent 60%),
            linear-gradient(var(--bg), var(--bg));
        color: var(--text);
        font-family: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
    }
    [data-testid="stAppViewContainer"]::before {
        content: ""; position: fixed; inset: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(43,245,160,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(43,245,160,0.035) 1px, transparent 1px);
        background-size: 28px 28px;
    }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden; height: 0;
    }
    .block-container { max-width: 1240px; padding: 1.6rem 1.4rem 3rem !important; }

    h1, h2, h3, h4, p, label, span, div { color: var(--text); }
    .mono { font-family: "JetBrains Mono", monospace; }

    /* ---------- Top bar ---------- */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        gap: 16px; flex-wrap: wrap;
        padding: 14px 18px; border: 1px solid var(--line); border-radius: 14px;
        background: rgba(14,21,19,0.8); backdrop-filter: blur(8px);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
        background: var(--pulse-dim); border: 1px solid rgba(43,245,160,0.35); font-size: 20px;
    }
    .brand-name { font-weight: 700; letter-spacing: -0.01em; font-size: 1.05rem; }
    .brand-sub { color: var(--muted); font-size: 0.75rem; font-family: "JetBrains Mono", monospace; }
    .status {
        font-family: "JetBrains Mono", monospace; font-size: 0.75rem; color: var(--muted);
        display: flex; align-items: center; gap: 8px;
    }
    .blink {
        width: 8px; height: 8px; border-radius: 50%; background: var(--pulse);
        box-shadow: 0 0 10px var(--pulse); animation: blink 1.2s infinite;
    }
    .blink.off { background: var(--alarm); box-shadow: 0 0 10px var(--alarm); }
    @keyframes blink { 50% { opacity: 0.25; } }

    /* ---------- ECG hero ---------- */
    .hero { margin: 22px 0 18px; }
    .hero h1 {
        font-size: clamp(2rem, 4.6vw, 3.6rem); line-height: 1; letter-spacing: -0.04em;
        font-weight: 700; margin: 0;
    }
    .hero h1 em { font-style: normal; color: var(--pulse); }
    .hero p { color: var(--muted); max-width: 560px; margin: 12px 0 0; }
    .ecg { width: 100%; height: 70px; margin-top: 10px; }
    .ecg path {
        fill: none; stroke: var(--pulse); stroke-width: 2;
        filter: drop-shadow(0 0 6px rgba(43,245,160,0.7));
        stroke-dasharray: 1400; stroke-dashoffset: 1400;
        animation: trace 3.2s linear infinite;
    }
    @keyframes trace { to { stroke-dashoffset: 0; } }

    /* ---------- Panels ---------- */
    .panel-label {
        font-family: "JetBrains Mono", monospace; font-size: 0.72rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--muted); margin: 0 0 10px;
        display: flex; justify-content: space-between;
    }
    div[data-testid="stForm"] {
        background: var(--panel); border: 1px solid var(--line) !important;
        border-radius: 18px; padding: 18px 20px 20px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: var(--panel-2); padding: 5px; border-radius: 12px;
        border: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px; padding: 0 16px; border-radius: 9px; background: transparent;
        font-family: "JetBrains Mono", monospace; font-size: 0.8rem; color: var(--muted);
    }
    .stTabs [aria-selected="true"] { background: var(--pulse-dim) !important; }
    .stTabs [aria-selected="true"] p { color: var(--pulse) !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Inputs */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important; color: var(--muted) !important; font-weight: 500;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    .stNumberInput input {
        background: var(--panel-2) !important; border-color: var(--line) !important;
        color: var(--text) !important; border-radius: 10px !important;
        font-family: "JetBrains Mono", monospace;
    }
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {
        border-color: var(--pulse) !important; box-shadow: 0 0 0 3px var(--pulse-dim) !important;
    }
    .stSlider [role="slider"] { background: var(--pulse) !important; box-shadow: 0 0 12px var(--pulse); }
    .stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--pulse) !important; font-family: "JetBrains Mono", monospace;
    }
    div[role="radiogroup"] { gap: 8px; }
    div[role="radiogroup"] label {
        background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px;
        padding: 6px 12px; margin: 0 !important;
    }
    div[role="radiogroup"] label:has(input:checked) {
        border-color: var(--pulse); background: var(--pulse-dim);
    }

    /* Submit */
    .stFormSubmitButton button {
        width: 100%; height: 52px; border-radius: 12px; border: none !important;
        background: var(--pulse) !important; color: #04130D !important;
        font-weight: 700; letter-spacing: 0.02em; font-size: 1rem;
        box-shadow: 0 0 0 0 rgba(43,245,160,0.5); transition: box-shadow .2s, transform .2s;
    }
    .stFormSubmitButton button p { color: #04130D !important; }
    .stFormSubmitButton button:hover {
        transform: translateY(-1px); box-shadow: 0 0 28px rgba(43,245,160,0.45);
    }

    /* ---------- Monitor ---------- */
    .monitor {
        background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
        padding: 20px; position: relative; overflow: hidden;
    }
    .monitor.high { border-color: rgba(255,77,94,0.55); box-shadow: 0 0 40px rgba(255,77,94,0.12); }
    .monitor.low  { border-color: rgba(43,245,160,0.45); box-shadow: 0 0 40px rgba(43,245,160,0.10); }
    .gauge-wrap { display: grid; place-items: center; padding: 8px 0 4px; }
    .gauge-num {
        font-family: "JetBrains Mono", monospace; font-size: 2.6rem; font-weight: 600;
        margin-top: -86px; line-height: 1;
    }
    .gauge-cap { color: var(--muted); font-size: 0.72rem; font-family: "JetBrains Mono", monospace;
                 letter-spacing: .12em; margin-top: 6px; text-transform: uppercase; }
    .verdict {
        text-align: center; margin: 30px 0 6px; font-size: 1.35rem; font-weight: 700;
        letter-spacing: -0.02em;
    }
    .verdict.high { color: var(--alarm); }
    .verdict.low  { color: var(--pulse); }
    .verdict-copy { text-align: center; color: var(--muted); font-size: 0.88rem; margin: 0 6px 14px; }
    .idle { text-align: center; padding: 34px 10px; color: var(--muted); }
    .idle-ring {
        width: 120px; height: 120px; margin: 0 auto 16px; border-radius: 50%;
        border: 2px dashed var(--line); display: grid; place-items: center; font-size: 42px;
        animation: beat 1.4s ease-in-out infinite;
    }
    @keyframes beat { 0%,100% { transform: scale(1); } 15% { transform: scale(1.08); } 30% { transform: scale(1); } }

    .vitals { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 14px; }
    .vital {
        background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px;
        padding: 10px 12px;
    }
    .vital-k { color: var(--muted); font-size: 0.68rem; font-family: "JetBrains Mono", monospace;
               text-transform: uppercase; letter-spacing: .1em; }
    .vital-v { font-family: "JetBrains Mono", monospace; font-size: 1.1rem; font-weight: 600; margin-top: 2px; }
    .vital-v small { color: var(--muted); font-size: 0.7rem; font-weight: 400; }

    .disclaimer {
        margin-top: 14px; padding: 10px 12px; border-radius: 10px;
        border: 1px solid rgba(255,194,75,0.3); background: rgba(255,194,75,0.07);
        color: var(--amber) !important; font-size: 0.78rem; line-height: 1.45;
    }
    .footer-note {
        text-align: center; color: var(--muted); font-family: "JetBrains Mono", monospace;
        font-size: 0.72rem; margin-top: 2.2rem;
    }
    details, [data-testid="stExpander"] {
        background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] input, [data-testid="stNumberInput"] button {
        background-color: var(--panel-2) !important; color: var(--text) !important;
    }
    [data-testid="stSelectbox"] svg { fill: var(--muted) !important; }
    .stTabs button[role="tab"] { border-bottom: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    [data-testid="stSlider"] div[role="slider"] { background-color: var(--pulse) !important; }
    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child > div {
        background: linear-gradient(to right, var(--pulse) 0%, var(--pulse) 100%) !important;
        filter: none;
    }
    div[role="radiogroup"] input + div { border-color: var(--pulse) !important; }
    /* ---------- v4: section cards, motion, insights ---------- */
    @keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
    @keyframes glow { 0%,100% { box-shadow: 0 0 30px rgba(255,77,94,0.10); } 50% { box-shadow: 0 0 60px rgba(255,77,94,0.28); } }
    @keyframes sweep { from { stroke-dashoffset: 283; } }
    @keyframes shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }
    .hero, .topbar { animation: rise .6s ease both; }
    .hero h1 em {
        background: linear-gradient(90deg, #2BF5A0, #9BFFD6, #2BF5A0); background-size: 200% 100%;
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        animation: shimmer 4s linear infinite;
    }
    .sec-head {
        display: flex; align-items: center; gap: 10px; margin: 18px 0 8px;
        animation: rise .5s ease both;
    }
    .sec-num {
        font-family: "JetBrains Mono", monospace; font-size: .72rem; color: var(--pulse);
        border: 1px solid rgba(43,245,160,.35); background: var(--pulse-dim);
        border-radius: 8px; padding: 3px 8px;
    }
    .sec-title { font-weight: 700; font-size: 1rem; letter-spacing: -.01em; }
    .sec-sub { color: var(--muted); font-size: .78rem; margin-left: auto; font-family: "JetBrains Mono", monospace; }
    .sec-line { height: 1px; background: linear-gradient(90deg, var(--line), transparent); margin-bottom: 6px; }
    [data-testid="stForm"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stForm"] [data-testid="stElementContainer"] { transition: transform .2s ease; }
    [data-testid="stForm"] [data-testid="stColumn"] > div {
        background: var(--panel-2); border: 1px solid var(--line); border-radius: 14px;
        padding: 12px 14px 6px; transition: border-color .25s, transform .25s, box-shadow .25s;
    }
    [data-testid="stForm"] [data-testid="stColumn"] > div:hover {
        border-color: rgba(43,245,160,.4); transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0,0,0,.35);
    }
    .monitor { animation: rise .6s ease both; }
    .monitor.high { animation: rise .6s ease both, glow 2.4s ease-in-out infinite; }
    .gauge-arc { animation: sweep 1.2s cubic-bezier(.2,.8,.2,1) both; }
    .ins-title { font-family: "JetBrains Mono", monospace; font-size: .72rem; letter-spacing: .14em;
                 text-transform: uppercase; color: var(--muted); margin: 18px 0 8px; }
    .factor {
        display: flex; align-items: center; gap: 10px; padding: 9px 12px; margin-bottom: 6px;
        border-radius: 10px; background: var(--panel-2); border: 1px solid var(--line);
        animation: rise .5s ease both;
    }
    .factor .dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; }
    .factor.bad .dot { background: var(--alarm); box-shadow: 0 0 8px var(--alarm); }
    .factor.good .dot { background: var(--pulse); box-shadow: 0 0 8px var(--pulse); }
    .factor .fk { font-size: .86rem; font-weight: 600; }
    .factor .fv { margin-left: auto; font-family: "JetBrains Mono", monospace; font-size: .78rem; color: var(--muted); }
    .tips { margin: 0; padding-left: 18px; color: var(--muted); font-size: .85rem; line-height: 1.6; }
    .tips li { color: var(--muted); }
    .score-bar { display: flex; gap: 4px; margin-top: 4px; }
    .score-bar span { flex: 1; height: 6px; border-radius: 3px; background: var(--line); }
    .score-bar span.on { background: var(--alarm); box-shadow: 0 0 6px var(--alarm); }
    @media (max-width: 640px) { .hero h1 { font-size: 2rem; } .vitals { grid-template-columns: 1fr 1fr; } }
    /* ---------- v5: centered hero, stacked layout, readable dropdowns ---------- */
    .hero { text-align: center; }
    .hero p { margin: 12px auto 0; }
    .ecg { max-width: 760px; display: block; margin: 10px auto 0; }
    .block-container { max-width: 1400px !important; width: 100% !important; padding-left: 3rem !important; padding-right: 3rem !important; }
    .gh-link { color: var(--pulse) !important; text-decoration: none !important; font-family: monospace; font-size: .85rem;
      border: 1px solid var(--pulse); padding: 6px 12px; border-radius: 8px; white-space: nowrap; }
    .gh-link:hover { background: rgba(34,197,94,.12); }
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > div { background: #0f1512 !important; color: #e6efe9 !important; }
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary svg { color: #e6efe9 !important; fill: #e6efe9 !important; }
    .inputs-tbl { width: 100%; border-collapse: collapse; font-family: monospace; }
    .inputs-tbl td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,.08); color: #e6efe9; background: transparent; }
    .inputs-tbl td:first-child { color: #8aa396; }
    .inputs-tbl td:last-child { text-align: right; color: var(--pulse); }
    .monitor { margin-top: 26px; }
    div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul, ul[role="listbox"] {
        background-color: #121C19 !important; border: 1px solid #1F2E2A !important;
    }
    ul[role="listbox"] li, div[data-baseweb="popover"] li {
        background-color: #121C19 !important; color: #E6F2EE !important;
    }
    ul[role="listbox"] li *, div[data-baseweb="popover"] li * { color: #E6F2EE !important; }
    ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
        background-color: rgba(43,245,160,0.16) !important;
    }
    ul[role="listbox"] li:hover *, ul[role="listbox"] li[aria-selected="true"] * { color: #2BF5A0 !important; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] input { color: #E6F2EE !important; }
    @media (min-width: 900px) { .vitals { grid-template-columns: repeat(3, 1fr); } }
    </style>
    """,
    )

# Header

status_html = (
    '<span class="blink"></span>MODEL ONLINE'
    if artifacts_loaded
    else '<span class="blink off"></span>MODEL OFFLINE'
)
html_block(
    f"""
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">🫀</div>
        <div>
          <div class="brand-name">Cardio Monitor</div>
          <div class="brand-sub">heart-risk · logistic regression</div>
        </div>
      </div>
      <div style="display:flex;gap:12px;align-items:center;">
        <div class="status">{status_html}</div>
        <a class="gh-link" href="https://github.com/priyam-10/heart-disease-prediction-ml" target="_blank">&lt;/&gt; View code on GitHub</a>
      </div>
    </div>
    <div class="hero">
      <h1>Know your <em>heart</em><br/>in seconds.</h1>
      <p>Enter eleven clinical measurements and get an instant heart-disease risk estimate.</p>
      <svg class="ecg" viewBox="0 0 1200 70" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0 40 L180 40 L200 40 L215 30 L230 40 L260 40 L275 52 L292 4 L310 64 L325 40
                 L360 40 L380 30 L400 40 L560 40 L580 40 L595 30 L610 40 L640 40 L655 52 L672 4
                 L690 64 L705 40 L740 40 L760 30 L780 40 L940 40 L960 40 L975 30 L990 40 L1020 40
                 L1035 52 L1052 4 L1070 64 L1085 40 L1120 40 L1140 30 L1160 40 L1200 40"/>
      </svg>
    </div>
    """,
    )

if not artifacts_loaded:
    st.error(
        "Model files could not be loaded. Make sure `model/LR_model_heart.pkl`, "
        "`model/scaler.pkl` and `model/columns.pkl` exist."
    )
    with st.expander("Technical details"):
        st.code(load_error or "")



# Layout: input console (left) + result monitor (right)

left = st.container()
right = st.container()

values: dict = {}

with left:
    st.markdown(
        '<div class="panel-label"><span>// Patient input</span><span>11 fields</span></div>',
        unsafe_allow_html=True,
    )

    def section(num: str, title: str, sub: str) -> None:
        html_block(
            f'''<div class="sec-head"><span class="sec-num">{num}</span>
            <span class="sec-title">{title}</span><span class="sec-sub">{sub}</span></div>
            <div class="sec-line"></div>'''
        )

    with st.form("health_details_form"):
        section("01", "👤 Profile", "who")
        c1, c2, c3 = st.columns([1.4, 1, 1.2])
        with c1:
            values["age"] = st.slider("🎂 Age (years)", 18, 100, 40, key="age")
        with c2:
            values["sex"] = st.radio("⚥ Sex", ["M", "F"], horizontal=True, key="sex")
        with c3:
            values["chest_pain"] = st.selectbox(
                "💓 Chest pain type", ["ATA", "NAP", "TA", "ASY"], key="chest_pain",
                help="ATA: atypical angina · NAP: non-anginal · TA: typical angina · ASY: asymptomatic",
            )

        section("02", "🩺 Vitals", "at rest")
        c1, c2, c3 = st.columns(3)
        with c1:
            values["resting_bp"] = st.number_input(
                "🩺 Resting BP (mm Hg)", min_value=80, max_value=200, value=120, key="resting_bp"
            )
        with c2:
            values["cholesterol"] = st.number_input(
                "🩸 Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, key="cholesterol"
            )
        with c3:
            values["fasting_bs"] = st.radio(
                "🍬 Fasting sugar > 120", [0, 1], horizontal=True,
                format_func=lambda x: "Yes" if x == 1 else "No", key="fasting_bs",
            )

        section("03", "📈 Cardiac tests", "ECG & exercise")
        c1, c2 = st.columns(2)
        with c1:
            values["max_hr"] = st.slider("❤️ Max heart rate (bpm)", 60, 220, 150, key="max_hr")
        with c2:
            values["oldpeak"] = st.slider(
                "📉 Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1, key="oldpeak"
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            values["resting_ecg"] = st.selectbox(
                "📈 Resting ECG", ["Normal", "ST", "LVH"], key="resting_ecg"
            )
        with c2:
            values["st_slope"] = st.selectbox("〽️ ST slope", ["Up", "Flat", "Down"], key="st_slope")
        with c3:
            values["exercise_angina"] = st.radio(
                "🏃 Exercise angina", ["Y", "N"], horizontal=True,
                format_func=lambda x: "Yes" if x == "Y" else "No", key="exercise_angina",
            )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        predict_clicked = st.form_submit_button(
            "▶  Run risk analysis", disabled=not artifacts_loaded
        )



# Prediction pipeline (UNCHANGED): raw inputs -> one-hot -> expected
# column order -> scaler.transform -> model.predict

def build_input_frame(v: dict) -> pd.DataFrame:
    raw_input = {
        "Age": v["age"],
        "RestingBP": v["resting_bp"],
        "Cholesterol": v["cholesterol"],
        "FastingBS": v["fasting_bs"],
        "MaxHR": v["max_hr"],
        "Oldpeak": v["oldpeak"],
        "Sex_" + v["sex"]: 1,
        "ChestPainType_" + v["chest_pain"]: 1,
        "RestingECG_" + v["resting_ecg"]: 1,
        "ExerciseAngina_" + v["exercise_angina"]: 1,
        "ST_Slope_" + v["st_slope"]: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[expected_columns]


def run_prediction(v: dict):
    input_df = build_input_frame(v)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            probability = float(model.predict_proba(scaled_input)[0][1])
        except Exception:  # noqa: BLE001
            probability = None
    return prediction, probability



# Result monitor rendering

def gauge_svg(p: float, color: str) -> str:
    """Semi-circular gauge, p in [0, 1]."""
    p = min(max(p, 0.0), 1.0)
    arc_len = 283  # ~ pi * 90
    return f"""
    <svg width="240" height="140" viewBox="0 0 240 140" aria-hidden="true">
      <path d="M30 120 A90 90 0 0 1 210 120" fill="none" stroke="#1F2E2A" stroke-width="16" stroke-linecap="round"/>
      <path d="M30 120 A90 90 0 0 1 210 120" fill="none" stroke="{color}" stroke-width="16"
            class="gauge-arc" stroke-linecap="round" stroke-dasharray="{arc_len}" stroke-dashoffset="{arc_len * (1 - p):.1f}"
            style="filter: drop-shadow(0 0 8px {color}); transition: stroke-dashoffset .8s ease;"/>
    </svg>
    """


def vitals_html(v: dict) -> str:
    items = [
        ("Age", v.get("age"), "yrs"),
        ("Resting BP", v.get("resting_bp"), "mmHg"),
        ("Cholesterol", v.get("cholesterol"), "mg/dL"),
        ("Max HR", v.get("max_hr"), "bpm"),
        ("Oldpeak", v.get("oldpeak"), "mm"),
        ("ST slope", v.get("st_slope"), ""),
    ]
    cells = "".join(
        f'<div class="vital"><div class="vital-k">{html.escape(k)}</div>'
        f'<div class="vital-v">{html.escape(str(val))} <small>{u}</small></div></div>'
        for k, val, u in items
    )
    return f'<div class="vitals">{cells}</div>'



def risk_factors(v: dict):
    """Simple clinical reference checks shown alongside the model output.
    These do NOT affect the prediction; they only explain the inputs."""
    pred_max_hr = 220 - v["age"]
    checks = [
        ("Blood pressure", v["resting_bp"] >= 140, f'{v["resting_bp"]} mmHg', "≥140 is high"),
        ("Cholesterol", v["cholesterol"] >= 240, f'{v["cholesterol"]} mg/dL', "≥240 is high"),
        ("Fasting blood sugar", v["fasting_bs"] == 1, "> 120" if v["fasting_bs"] else "normal", ""),
        ("Max heart rate", v["max_hr"] < 0.7 * pred_max_hr, f'{v["max_hr"]} / {pred_max_hr} bpm', ""),
        ("ST depression", v["oldpeak"] >= 2.0, f'{v["oldpeak"]}', "≥2.0 is notable"),
        ("ST slope", v["st_slope"] in ("Flat", "Down"), v["st_slope"], ""),
        ("Exercise angina", v["exercise_angina"] == "Y", "Yes" if v["exercise_angina"] == "Y" else "No", ""),
        ("Chest pain type", v["chest_pain"] == "ASY", v["chest_pain"], ""),
        ("Age", v["age"] >= 55, f'{v["age"]} yrs', ""),
    ]
    return checks


def insights_html(v: dict, high: bool) -> str:
    if not v:
        return ""
    checks = risk_factors(v)
    flagged = [c for c in checks if c[1]]
    bar = "".join(f'<span class="{"on" if i < len(flagged) else ""}"></span>' for i in range(len(checks)))
    rows = "".join(
        f'<div class="factor {"bad" if bad else "good"}" style="animation-delay:{i*60}ms">'
        f'<span class="dot"></span><span class="fk">{html.escape(name)}</span>'
        f'<span class="fv">{html.escape(str(val))}</span></div>'
        for i, (name, bad, val, _) in enumerate(sorted(checks, key=lambda c: not c[1]))
    )
    tips = (
        ["Book a cardiology consultation soon.", "Ask about an ECG stress test or echocardiogram.",
         "Track blood pressure and cholesterol regularly.", "Avoid smoking and limit salt and alcohol."]
        if high else
        ["Keep up 150 min of moderate exercise a week.", "Eat plenty of vegetables, fibre and whole grains.",
         "Recheck blood pressure and cholesterol yearly.", "Sleep 7–8 hours and manage stress."]
    )
    tip_items = "".join(f"<li>{t}</li>" for t in tips)
    return (
        f'<div class="ins-title">Risk flags · {len(flagged)} of {len(checks)}</div>'
        f'<div class="score-bar">{bar}</div><div style="height:10px"></div>{rows}'
        f'<div class="ins-title">{"Next steps" if high else "Keep it up"}</div>'
        f'<ul class="tips">{tip_items}</ul>'
    )


def render_result(prediction, probability, inputs: dict) -> None:
    high = prediction == 1
    tone = "high" if high else "low"
    color = "#FF4D5E" if high else "#2BF5A0"
    title = "High risk of heart disease" if high else "Low risk of heart disease"
    copy = (
        "The model detected patterns associated with heart disease. "
        "Please consult a cardiologist for a proper evaluation."
        if high
        else "The model did not detect strong indicators of heart disease. "
        "Keep up regular check-ups and a heart-healthy lifestyle."
    )

    if probability is not None:
        gauge = gauge_svg(probability, color)
        num = f"{probability * 100:.0f}<span style='font-size:1.1rem;color:#7F9A92'>%</span>"
        cap = "estimated risk"
    else:
        gauge = gauge_svg(1.0 if high else 0.15, color)
        num = "HIGH" if high else "LOW"
        cap = "model verdict"

    html_block(
        f"""
        <div class="monitor {tone}" role="status">
          <div class="panel-label"><span>// Result</span><span>{'⚠ ALERT' if high else '✓ STABLE'}</span></div>
          <div class="gauge-wrap">
            {gauge}
            <div class="gauge-num" style="color:{color}">{num}</div>
            <div class="gauge-cap">{cap}</div>
          </div>
          <div class="verdict {tone}">{title}</div>
          <p class="verdict-copy">{copy}</p>
          {vitals_html(inputs)}
          {insights_html(inputs, high)}
          <div class="disclaimer">
            ⚕ This tool is for educational purposes only and is not a medical diagnosis.
            Always consult a qualified healthcare professional.
          </div>
        </div>
        """,
            )


def render_idle() -> None:
    html_block(
        """
        <div class="monitor">
          <div class="panel-label"><span>// Result</span><span>STANDBY</span></div>
          <div class="idle">
            <div class="idle-ring">🫀</div>
            <div style="font-weight:600;color:#E6F2EE;margin-bottom:6px">Awaiting input</div>
            <div style="font-size:.86rem">Fill in the three tabs and press
            <b style="color:#2BF5A0">Run risk analysis</b> to see the result here.</div>
          </div>
          <div class="disclaimer">
            ⚕ Educational tool only — not a substitute for professional medical advice.
          </div>
        </div>
        """,
            )


# -------------------------------------------------------------------
# Run + display
# -------------------------------------------------------------------
with right:
    if predict_clicked:
        try:
            prediction, probability = run_prediction(values)
            st.session_state["last_result"] = (prediction, probability)
            st.session_state["last_inputs"] = dict(values)
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("last_result", None)
            st.error(
                "Prediction could not be completed. Check that the model, scaler, "
                "feature columns, and preprocessing artifacts match one another."
            )
            with st.expander("Technical details"):
                st.code(str(exc))

    if "last_result" in st.session_state:
        render_result(*st.session_state["last_result"], st.session_state.get("last_inputs", {}))
        with st.expander("Inputs used for this prediction"):
            _rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in st.session_state.get("last_inputs", {}).items()
            )
            st.markdown(f'<table class="inputs-tbl">{_rows}</table>', unsafe_allow_html=True)
    else:
        render_idle()


st.markdown(
    '<div class="footer-note">cardio-monitor v6 · predictions generated by your configured model</div>',
    unsafe_allow_html=True,
)
