"""
Fake News Detection System - Streamlit Web Application

Interactive interface for analysing news text with a TF-IDF + Logistic
Regression pipeline. The prediction is a style-based classification, not a
fact check, and the UI says so wherever a result is shown.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Configuration (set_page_config must be the first Streamlit call)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "fake_news_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

APP_TITLE = "📰 Fake News Detection System"
PAGES = ["Home", "Analyze News", "Model Information", "About"]
MAX_CHARS = 20_000

SAMPLES = {
    "Real-style sample": (
        "WASHINGTON (Reuters) - The Senate approved legislation on Wednesday to reform "
        "bipartisan infrastructure spending, directing federal allocations toward highway "
        "repairs, clean drinking water initiatives, and high-speed broadband expansion "
        "across rural districts."
    ),
    "Fake-style sample": (
        "BREAKING EXPLOSIVE BOMBSHELL: Shocking secret military files exposed! Global elites "
        "secretly met in an underground bunker to overthrow the government and hide the "
        "ultimate truth from citizens. Share this before it gets deleted everywhere!"
    ),
}

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Source+Serif+4:wght@400;600&display=swap');
:root {
  --paper:#f4efe4; --card:#fbf8f1; --ink:#1c1a17; --muted:#6b645a; --rule:#1c1a17; --line:#d8d0bd;
  --accent:#9b1c1c;
  --real:#2f5d3a; --real-bg:#eef3ea; --real-line:#c3d4bc;
  --fake:#9b1c1c; --fake-bg:#f8e9e5; --fake-line:#e3bdb5;
  --serif:'Playfair Display', Georgia, 'Times New Roman', serif;
}
#MainMenu, footer {visibility:hidden;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] {display:none !important;}
.stApp {background:var(--paper);}
.block-container {max-width:1100px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3 {font-family:var(--serif) !important; color:var(--ink); letter-spacing:-0.01em;}
h4 {color:var(--ink);}
h2 {border-bottom:1px solid var(--line); padding-bottom:.35rem;}

/* Masthead */
.masthead {text-align:center; border-top:4px solid var(--rule); border-bottom:1px solid var(--rule);
  padding:.9rem 0 .7rem 0; margin-bottom:1.4rem; position:relative;}
.masthead::after {content:""; position:absolute; left:0; right:0; bottom:-6px; border-bottom:1px solid var(--rule);}
.masthead .name {font-family:var(--serif); font-weight:900; font-size:2.6rem; line-height:1.1; color:var(--ink);}
.masthead .tag {font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); margin-top:.25rem;}

/* Sidebar */
section[data-testid="stSidebar"] {background:#1c1a17;}
section[data-testid="stSidebar"] * {color:#e9e2d0;}
section[data-testid="stSidebar"] hr {border-color:#3a362f;}
.brand {font-family:var(--serif); font-size:1.25rem; font-weight:800; color:#fff !important;}
.brand-sub {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#a39a86 !important; margin-bottom:1.2rem;}
.status-pill {display:inline-block; padding:.25rem .7rem; border-radius:3px; font-size:.75rem;
  font-weight:700; letter-spacing:.06em; text-transform:uppercase;}
.status-ok {background:#2f5d3a; color:#fff !important;}
.status-no {background:#8a6d1f; color:#fff !important;}
.side-note {font-size:.78rem; color:#a39a86 !important; line-height:1.5; margin-top:1rem;}

/* Lead story (hero) */
.hero {border-top:3px solid var(--rule); border-bottom:1px solid var(--line); padding:1.4rem 0 1.5rem 0; margin-bottom:1.6rem;}
.kicker {color:var(--accent); font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase;}
.hero h1 {font-size:2.5rem; line-height:1.15; margin:.35rem 0 .6rem 0; font-weight:800;}
.hero p {font-family:'Source Serif 4', Georgia, serif; color:#3b372f; font-size:1.12rem; line-height:1.6; margin:0 0 1rem 0; max-width:680px;}
.chip {display:inline-block; border:1px solid var(--ink); color:var(--ink); border-radius:2px;
  padding:.15rem .6rem; font-size:.74rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin-right:.4rem;}

/* Cards */
.card {background:var(--card); border:1px solid var(--line); border-top:3px solid var(--ink);
  border-radius:2px; padding:1.1rem 1.2rem; min-height:140px;}
.card h4 {margin:0 0 .4rem 0; font-family:var(--serif); font-size:1.08rem;}
.card p {margin:0; color:var(--muted); font-size:.92rem; line-height:1.55;}
.step-num {font-family:var(--serif); font-size:2rem; font-weight:900; color:var(--accent); line-height:1; margin-bottom:.3rem;}

/* Stat tiles */
.stat {background:var(--card); border:1px solid var(--line); border-top:3px solid var(--ink); padding:1rem 1.1rem; height:100%;}
.stat .n {font-family:var(--serif); font-size:2.1rem; font-weight:800; line-height:1.1;}
.stat .l {font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-top:.3rem;}
.stat .d {font-size:.84rem; color:var(--muted); line-height:1.45; margin-top:.4rem;}

/* Result */
.verdict {border-radius:2px; padding:1.3rem 1.5rem; border:1px solid; border-left-width:6px;}
.verdict.real {background:var(--real-bg); border-color:var(--real-line); border-left-color:var(--real);}
.verdict.fake {background:var(--fake-bg); border-color:var(--fake-line); border-left-color:var(--fake);}
.verdict .label {font-size:.74rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; color:var(--muted);}
.verdict .big {font-family:var(--serif); font-size:2.1rem; font-weight:900; margin:.15rem 0 .4rem 0;}
.verdict.real .big {color:var(--real);} .verdict.fake .big {color:var(--fake);}
.verdict p {margin:0; color:#3b372f; font-size:.95rem; line-height:1.55;}
.bar-row {margin:.55rem 0;}
.bar-head {display:flex; justify-content:space-between; font-size:.88rem; font-weight:600; margin-bottom:.25rem;}
.bar-track {background:#e2dac7; height:10px; overflow:hidden;}
.bar-fill {height:100%;}
.bar-real {background:var(--real);} .bar-fake {background:var(--fake);}
.term {display:inline-block; padding:.18rem .6rem; margin:.15rem .25rem .15rem 0; font-size:.85rem;
  font-weight:600; border:1px solid; border-radius:2px;}
.term.f {background:var(--fake-bg); border-color:var(--fake-line); color:var(--fake);}
.term.r {background:var(--real-bg); border-color:var(--real-line); color:var(--real);}
.disclaimer {background:#f1e8cc; border:1px solid #dccb94; border-left:5px solid #a9801a; padding:.9rem 1.1rem;
  color:#4d3d0f; font-size:.9rem; line-height:1.55;}

/* Model page */
.split {display:flex; height:26px; overflow:hidden; border:1px solid var(--ink);}
.split div {display:flex; align-items:center; justify-content:center; color:#fff; font-size:.78rem; font-weight:700;}
.pipe {display:flex; gap:.6rem; align-items:stretch; flex-wrap:wrap;}
.pipe .box {flex:1; min-width:160px; background:var(--card); border:1px solid var(--ink); padding:.8rem 1rem;}
.pipe .arrow {align-self:center; font-size:1.4rem; color:var(--muted);}
.pipe .box b {font-family:var(--serif);} .pipe .box span {display:block; color:var(--muted); font-size:.84rem; margin-top:.2rem;}
.kv {width:100%; border-collapse:collapse; font-size:.92rem;}
.kv td {padding:.5rem .3rem; border-bottom:1px solid var(--line);} .kv td:first-child {color:var(--muted); width:55%;}
.kv td:last-child {text-align:right; font-weight:600;}

/* Native widgets */
.stTextArea textarea {border-radius:2px; border:1px solid #b9b09b; background:#fffdf8; font-size:.98rem;}
.stButton > button {border-radius:2px; min-height:44px; font-weight:600; border:1px solid var(--ink);
  background:var(--card); color:var(--ink);}
.stButton > button:hover {border-color:var(--accent); color:var(--accent);}
.stButton > button[kind="primary"] {background:var(--ink); border:1px solid var(--ink); color:#fff;}
.stButton > button[kind="primary"]:hover {background:var(--accent); border-color:var(--accent); color:#fff;}

/* Top navigation (replaces the tagline row) */
.st-key-topnav {border-top:4px solid var(--rule); border-bottom:1px solid var(--rule);
  padding:.55rem 0 .45rem 0; margin-bottom:1.4rem; position:relative;}
.st-key-topnav::after {content:""; position:absolute; left:0; right:0; bottom:-6px; border-bottom:1px solid var(--rule);}
.st-key-topnav div[role="radiogroup"] {justify-content:center; flex-wrap:wrap; gap:.4rem 2rem;}
.st-key-topnav label {padding:.25rem 0 !important; margin:0 !important; cursor:pointer;
  border-bottom:2px solid transparent;}
.st-key-topnav label > div:first-child {display:none;}
.st-key-topnav label p {font-size:.8rem !important; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink);}
.st-key-topnav label:hover p {color:var(--accent);}
.st-key-topnav label:has(input:checked) {border-bottom-color:var(--accent);}
.st-key-topnav label:has(input:checked) p {color:var(--accent);}
@media (max-width:640px) {.st-key-topnav div[role="radiogroup"] {gap:.3rem 1.2rem;}
  .st-key-topnav label p {font-size:.72rem !important; letter-spacing:.1em;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_cached_model():
    """Load the trained pipeline once per server process."""
    if not MODEL_PATH.exists():
        return None
    try:
        from src.predict import load_model_pipeline
        return load_model_pipeline(MODEL_PATH)
    except Exception as exc:  # surfaced in the UI instead of crashing
        st.session_state["model_error"] = str(exc)
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def get_model_info() -> Dict[str, Any]:
    """Return metrics and training info from the evaluation report or model metadata."""
    report = _read_json(METRICS_PATH) or {}
    meta = _read_json(METADATA_PATH) or {}
    metrics = report.get("metrics") or meta.get("metrics") or {}
    return {
        "metrics": metrics,
        "test_samples": report.get("total_test_samples") or meta.get("test_samples"),
        "train_samples": meta.get("train_samples"),
        "dataset": meta.get("dataset_summary") or {},
        "params": meta.get("hyperparameters") or {},
        "saved_at": meta.get("saved_at") or report.get("evaluation_timestamp"),
        "raw": {"report": report, "metadata": meta},
    }


def top_terms(pipeline, cleaned_text: str, n: int = 6) -> Dict[str, List[tuple]]:
    """Words in the text that pushed the model toward Fake or Real (linear model weights)."""
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["classifier"]
        vec = tfidf.transform([cleaned_text])
        names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]
        contribs = [(names[i], float(vec[0, i] * coefs[i])) for i in vec.nonzero()[1]]
    except Exception:
        return {"fake": [], "real": []}
    fake = sorted([c for c in contribs if c[1] > 0], key=lambda c: -c[1])[:n]
    real = sorted([c for c in contribs if c[1] < 0], key=lambda c: c[1])[:n]
    return {"fake": fake, "real": real}


def prob_bar(label: str, value: float, css: str) -> str:
    pct = value * 100
    return (
        f'<div class="bar-row"><div class="bar-head"><span>{label}</span>'
        f"<span>{pct:.1f}%</span></div>"
        f'<div class="bar-track"><div class="bar-fill {css}" style="width:{pct:.1f}%"></div></div></div>'
    )


def terms_html(items: List[tuple], css: str) -> str:
    if not items:
        return '<span style="color:#64748b;font-size:.88rem">None found in this text.</span>'
    return "".join(f'<span class="term {css}">{html.escape(t)}</span>' for t, _ in items)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
model_available = MODEL_PATH.exists()

if st.session_state.pop("goto_analyze", False):
    st.session_state["page"] = "Analyze News"

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def masthead() -> None:
    st.title(APP_TITLE)
    with st.container(key="topnav"):
        st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")


def page_home() -> None:
    masthead()
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Lead story · How it works</div>
          <h1>Spot suspicious writing patterns in news</h1>
          <p>Paste a headline or article and get an instant, explainable estimate of whether
          its style resembles real reporting or fake news from the training data.</p>
          <span class="chip">Natural Language Processing</span>
          <span class="chip">TF-IDF</span><span class="chip">Logistic Regression</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    cards = [
        ("🔍 Analyze text", "Paste a headline or full article and see the prediction with confidence scores."),
        ("💡 See why", "Review the words that pushed the model toward Real or Fake."),
        ("📊 Inspect the model", "Check accuracy, precision, recall and the data it was trained on."),
    ]
    for col, (title, body) in zip((c1, c2, c3), cards):
        col.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("Start analyzing  →", type="primary"):
        st.session_state["goto_analyze"] = True
        st.rerun()

    st.subheader("How it works")
    s1, s2, s3 = st.columns(3)
    steps = [
        ("1", "Input", "You provide a headline or article body."),
        ("2", "Vectorize", "TF-IDF turns the text into weighted word and phrase features."),
        ("3", "Classify", "Logistic Regression scores the text as Real or Fake."),
    ]
    for col, (num, title, body) in zip((s1, s2, s3), steps):
        col.markdown(
            f'<div class="card"><div class="step-num">{num}</div><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div class="disclaimer"><b>Important:</b> this system analyses text patterns. It does not '
        "verify whether a story is true. Predictions can be wrong, so check important claims against "
        "multiple reliable sources.</div>",
        unsafe_allow_html=True,
    )


def page_analyze() -> None:
    masthead()
    st.header("Analyze a news article")

    pipeline = get_cached_model()
    if not model_available or pipeline is None:
        st.warning("The trained model is not available yet.")
        if st.session_state.get("model_error"):
            st.error(st.session_state["model_error"])
        st.markdown(
            f"""
            To train it:
            1. Download `Fake.csv` and `True.csv` from the
               [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
            2. Place both files in the `data/` folder.
            3. Run `python -m src.train_model`, then refresh this page.

            Expected model file: `models/{MODEL_PATH.name}`
            """
        )
        return

    # Sample buttons use callbacks so they can update the text box's state safely.
    def set_sample(key: str) -> None:
        st.session_state["news_input"] = SAMPLES[key]

    def clear_text() -> None:
        st.session_state["news_input"] = ""

    st.caption("Try an example or paste your own text.")
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 3])
    b1.button("Real-style sample", on_click=set_sample, args=("Real-style sample",), use_container_width=True)
    b2.button("Fake-style sample", on_click=set_sample, args=("Fake-style sample",), use_container_width=True)
    b3.button("Clear", on_click=clear_text, use_container_width=True)

    text = st.text_area(
        "News headline or article text",
        key="news_input",
        height=200,
        max_chars=MAX_CHARS,
        placeholder="Paste a headline and article body here...",
    )
    stripped = text.strip()
    words = len(stripped.split()) if stripped else 0
    st.caption(f"{len(stripped):,} characters · {words:,} words"
               + ("  ·  Short text gives less reliable results." if 0 < words < 20 else ""))

    if not st.button("Analyze", type="primary"):
        return
    if not stripped:
        st.error("Please enter some text to analyze.")
        return

    from src.predict import predict_news
    try:
        with st.spinner("Analyzing..."):
            result = predict_news(text, model=pipeline)
    except Exception as err:
        st.error(f"Prediction failed: {err}")
        return

    is_fake = result.predicted_label == 1
    probs = result.probabilities or {}
    real_p, fake_p = probs.get("Real", 0.0), probs.get("Fake", 0.0)
    confidence = max(real_p, fake_p)

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        verdict_cls = "fake" if is_fake else "real"
        headline = "Likely Fake" if is_fake else "Likely Real"
        blurb = (
            "The vocabulary and style closely resemble articles labeled as fake news in the training data."
            if is_fake else
            "The vocabulary and style closely resemble articles labeled as real news in the training data."
        )
        st.markdown(
            f'<div class="verdict {verdict_cls}"><div class="label">Prediction</div>'
            f'<div class="big">{headline}</div><p>{blurb}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        if probs:
            st.markdown(
                prob_bar("Real similarity", real_p, "bar-real") + prob_bar("Fake similarity", fake_p, "bar-fake"),
                unsafe_allow_html=True,
            )
            level = "High" if confidence >= 0.9 else "Moderate" if confidence >= 0.7 else "Low"
            st.caption(f"Model confidence: **{level}** ({confidence * 100:.1f}%). "
                       "Scores measure style similarity, not truth.")

    terms = top_terms(pipeline, result.cleaned_text)
    if terms["fake"] or terms["real"]:
        st.subheader("What influenced this result")
        t1, t2 = st.columns(2)
        t1.markdown("**Words pushing toward Fake**", help="Highest-weighted terms found in your text.")
        t1.markdown(terms_html(terms["fake"], "f"), unsafe_allow_html=True)
        t2.markdown("**Words pushing toward Real**")
        t2.markdown(terms_html(terms["real"], "r"), unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div class="disclaimer"><b>Verification notice:</b> this is an automated style classification. '
        "It cannot confirm whether an event happened. Consult multiple independent sources "
        "for anything important.</div>",
        unsafe_allow_html=True,
    )


def stat_tile(value: str, label: str, desc: str = "") -> str:
    d = f'<div class="d">{desc}</div>' if desc else ""
    return f'<div class="stat"><div class="n">{value}</div><div class="l">{label}</div>{d}</div>'


@st.cache_data(show_spinner=False)
def global_top_terms(_pipeline, n: int = 12) -> Dict[str, List[str]]:
    """Most indicative words overall, taken from the classifier weights."""
    try:
        names = _pipeline.named_steps["tfidf"].get_feature_names_out()
        coefs = _pipeline.named_steps["classifier"].coef_[0]
    except Exception:
        return {"fake": [], "real": []}
    order = coefs.argsort()
    return {"fake": [names[i] for i in order[::-1][:n]], "real": [names[i] for i in order[:n]]}


def page_model() -> None:
    masthead()
    st.header("Model information")
    info = get_model_info()
    m, ds, params = info["metrics"], info["dataset"], info["params"]

    # --- Performance
    st.subheader("Performance on unseen articles")
    if not m:
        st.info("No evaluation metrics found yet. Run `python -m src.train_model` to generate them.")
    else:
        tiles = [
            ("accuracy", "Accuracy", "Share of all test articles classified correctly."),
            ("precision_fake", "Precision (Fake)", "When it says Fake, how often it is right."),
            ("recall_fake", "Recall (Fake)", "Share of all fake articles it catches."),
            ("f1_score_fake", "F1 score (Fake)", "Balance of precision and recall."),
        ]
        for col, (key, label, desc) in zip(st.columns(4), tiles):
            col.markdown(stat_tile(f"{m.get(key, 0.0) * 100:.1f}%", label, desc), unsafe_allow_html=True)
        bits = []
        if info["test_samples"]:
            bits.append(f"Measured on {info['test_samples']:,} held-out test articles")
        if info["saved_at"]:
            bits.append(f"trained {str(info['saved_at'])[:10]}")
        if bits:
            st.caption(" · ".join(bits) + ".")
        st.markdown(
            '<div class="disclaimer"><b>Read these numbers with care.</b> Scores this high usually mean the '
            "dataset has easy giveaways (for example, real articles carrying a news-agency tag). Accuracy on "
            "other outlets and newer topics is likely lower.</div>",
            unsafe_allow_html=True,
        )

    if CONFUSION_MATRIX_PATH.exists():
        st.subheader("Confusion matrix")
        st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)

    # --- Data
    if ds:
        st.subheader("Training data")
        real, fake = ds.get("real_count", 0), ds.get("fake_count", 0)
        total = ds.get("total_samples", real + fake) or 1
        rp, fp = real / total * 100, fake / total * 100
        st.markdown(
            f'<div class="split"><div style="width:{rp:.1f}%;background:#2f5d3a">Real {real:,} ({rp:.0f}%)</div>'
            f'<div style="width:{fp:.1f}%;background:#9b1c1c">Fake {fake:,} ({fp:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        tiles = [
            (f"{total:,}", "Articles used", "After cleaning."),
            (f"{info['train_samples']:,}" if info["train_samples"] else "n/a", "Training set", "Used to fit the model."),
            (f"{info['test_samples']:,}" if info["test_samples"] else "n/a", "Test set", "Held out for evaluation."),
            (f"{ds.get('duplicate_articles_removed', 0):,}", "Duplicates removed", "Prevents test-set leakage."),
        ]
        for col, (v, l, d) in zip(st.columns(4), tiles):
            col.markdown(stat_tile(v, l, d), unsafe_allow_html=True)

    # --- Pipeline
    st.subheader("How the model works")
    ngram = params.get("ngram_range")
    ngram_txt = f"{ngram[0]}-{ngram[1]} word phrases" if ngram else "word phrases"
    feats = f"up to {params['max_features']:,} features" if params.get("max_features") else "weighted features"
    st.markdown(
        f'''<div class="pipe">
        <div class="box"><b>1. Clean text</b><span>Normalize spacing and case.</span></div><div class="arrow">→</div>
        <div class="box"><b>2. TF-IDF</b><span>{ngram_txt}, {feats}, English stop words removed.</span></div><div class="arrow">→</div>
        <div class="box"><b>3. Logistic Regression</b><span>L2 regularization, balanced class weights.</span></div><div class="arrow">→</div>
        <div class="box"><b>4. Result</b><span>Real or Fake with a probability.</span></div></div>''',
        unsafe_allow_html=True,
    )

    # --- Learned vocabulary
    pipeline = get_cached_model()
    if pipeline is not None:
        terms = global_top_terms(pipeline)
        if terms["fake"]:
            st.subheader("Most telling words")
            st.caption("Words the model weighs most heavily overall. These are patterns in this dataset, not proof of falsehood.")
            c1, c2 = st.columns(2)
            c1.markdown("**Signal Fake**")
            c1.markdown("".join(f'<span class="term f">{html.escape(t)}</span>' for t in terms["fake"]), unsafe_allow_html=True)
            c2.markdown("**Signal Real**")
            c2.markdown("".join(f'<span class="term r">{html.escape(t)}</span>' for t in terms["real"]), unsafe_allow_html=True)

    # --- Settings
    if params:
        st.subheader("Training settings")
        labels = {"max_features": "Maximum vocabulary size", "ngram_range": "Phrase length (words)",
                  "test_size": "Share held out for testing", "random_state": "Random seed"}
        rows = []
        for k, label in labels.items():
            if k in params:
                v = params[k]
                v = f"{v[0]} to {v[1]}" if k == "ngram_range" else f"{v:.0%}" if k == "test_size" else f"{v:,}" if k == "max_features" else v
                rows.append(f"<tr><td>{label}</td><td>{v}</td></tr>")
        st.markdown(f'<table class="kv">{"".join(rows)}</table>', unsafe_allow_html=True)


def page_about() -> None:
    masthead()
    st.header("About this project")
    st.markdown(
        "Online misinformation is hard to spot at scale. This project uses classic NLP to highlight "
        "linguistic markers (sensational wording, hyperbole, structural cues) often seen in misleading articles."
    )
    a, b = st.columns(2)
    a.markdown(
        '<div class="card"><h4>Technology stack</h4><p>Python · pandas · NumPy · scikit-learn · '
        "Streamlit · Matplotlib · pytest · Streamlit Community Cloud</p></div>",
        unsafe_allow_html=True,
    )
    b.markdown(
        '<div class="card"><h4>Future work</h4><p>Claim verification, multi-domain training data, '
        "SHAP/LIME explanations and probability calibration.</p></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Known limitations")
    st.markdown(
        """
        - **Domain and time shift:** a model trained on past political news may not transfer to new topics.
        - **Satire:** parody uses exaggeration for humor and is often misclassified.
        - **Polished misinformation:** false stories written in formal journalistic prose can look real to a style model.
        - **Short text:** a single sentence or tweet has too little vocabulary for a confident result.
        """
    )


page = st.session_state.get("page", "Home")
{"Home": page_home, "Analyze News": page_analyze, "Model Information": page_model, "About": page_about}[page]()
