"""
Fake News Detection System - Streamlit Web Application

An interactive web application for analyzing news articles using a trained
machine learning pipeline with TF-IDF feature extraction and Logistic Regression.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st

# Application Configuration (must be called first)
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative Path Constants (compatible with local execution and Streamlit Community Cloud)
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = MODELS_DIR / "fake_news_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics_summary.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"


@st.cache_resource(show_spinner="Loading trained model pipeline...")
def get_cached_model():
    """Caches the model pipeline in memory for high-performance inference."""
    if not MODEL_PATH.exists():
        return None
    try:
        from src.predict import load_model_pipeline
        return load_model_pipeline(MODEL_PATH)
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def get_latest_metrics() -> Optional[Dict[str, Any]]:
    """Loads metrics report if available."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sidebar Navigation & Status
st.sidebar.title("📰 Navigation")
page = st.sidebar.radio(
    "Select a Section:",
    ["🏠 Home", "🔍 News Analysis", "📊 Model Information", "ℹ️ About Project"],
)

model_available = MODEL_PATH.exists()
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if model_available:
    st.sidebar.success("🟢 Model: Trained & Available")
else:
    st.sidebar.warning("🟡 Model: Not Trained Yet")

st.sidebar.info(
    "**Notice:** Automated machine learning text analysis evaluates lexical and "
    "stylistic patterns. It does not replace independent fact-checking."
)


# -----------------------------------------------------------------------------
# PAGE A: HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("📰 Fake News Detection System")
    st.subheader("Machine Learning-Based News Text Classification")

    st.markdown("""
    Welcome to the **Fake News Detection System**, an educational machine learning application 
    designed to identify lexical, stylistic, and linguistic markers commonly associated with 
    unreliable or artificially fabricated news text.

    ---
    ### ⚙️ How It Works
    1. **Text Preprocessing:** Cleans input by normalizing whitespace, standardizing casing, and eliminating noisy characters.
    2. **TF-IDF Feature Extraction:** Converts textual articles into statistical n-gram vectors (Term Frequency - Inverse Document Frequency) emphasizing distinctive vocabulary patterns.
    3. **Logistic Regression Classification:** A lightweight, interpretable linear classifier calculates class probability boundaries learned from labeled historical datasets.
    4. **Safety & Transparency:** Returns transparent model score estimates accompanied by essential context regarding algorithmic limitations.

    ---
    ### ⚠️ Crucial Ethical & Operational Disclaimer
    > **Machine learning models classify writing style, not objective reality.**  
    > An algorithm trained on text patterns cannot independently confirm whether a real-world event happened. 
    > A fabricated claim written in formal journalistic tone might deceive the model, while a factual breaking story 
    > written in an informal tone might be flagged.  
    > **Always verify critical claims through multiple authoritative, independent primary sources.**
    """)

    st.markdown("### 🚀 Quick Start Guide")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. For Users:**
        - Navigate to **🔍 News Analysis** in the left sidebar.
        - Paste a news article or headline into the analysis box.
        - Click **Analyze News** to evaluate the stylistic markers.
        """)
    with col2:
        st.markdown("""
        **2. For Developers:**
        - Inspect training metrics in **📊 Model Information**.
        - Train or retrain models locally with `python -m src.train_model`.
        - Run automated software tests via `python -m pytest tests/ -v`.
        """)


# -----------------------------------------------------------------------------
# PAGE B: NEWS ANALYSIS
# -----------------------------------------------------------------------------
elif page == "🔍 News Analysis":
    st.title("🔍 News Article Analysis")
    st.markdown("Enter a news headline or complete article text below to analyze its linguistic patterns.")

    # Check model readiness
    pipeline = get_cached_model()

    if not model_available or pipeline is None:
        st.warning("⚠️ **Trained Model Artifact Not Found**")
        st.markdown(f"""
        The model file `{MODEL_PATH.name}` has not yet been generated in `models/`.
        
        To train the model:
        1. Download the **Fake and Real News Dataset** (`Fake.csv` and `True.csv`) from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
        2. Place both CSV files in the `data/` directory.
        3. Execute the training command in your terminal:
        ```bash
        python -m src.train_model
        ```
        4. Refresh this page once training completes.
        """)
    else:
        # Sample text buttons for user convenience
        st.markdown("##### Quick Test Samples")
        col_s1, col_s2, col_s3 = st.columns([1, 1, 3])
        sample_text = ""
        if col_s1.button("Load Real Sample"):
            st.session_state["news_input"] = (
                "WASHINGTON (Reuters) - The Senate approved legislation on Wednesday to reform "
                "bipartisan infrastructure spending, directing federal allocations toward highway repairs, "
                "clean drinking water initiatives, and high-speed broadband expansion across rural districts."
            )
        if col_s2.button("Load Fake Sample"):
            st.session_state["news_input"] = (
                "BREAKING EXPLOSIVE BOMBSHELL: Shocking secret military files exposed! "
                "Global elites secretly met in an underground bunker to overthrow the government "
                "and hide the ultimate truth from citizens. Share this before it gets deleted everywhere!"
            )

        news_text = st.text_area(
            "Paste News Headline or Article Text:",
            value=st.session_state.get("news_input", ""),
            height=200,
            placeholder="Paste headline and article body here...",
            key="news_input_area",
        )

        char_count = len(news_text.strip())
        word_count = len(news_text.strip().split()) if char_count > 0 else 0
        st.caption(f"Input statistics: {char_count:,} characters | {word_count:,} words")

        if st.button("🚀 Analyze News", type="primary"):
            if not news_text.strip():
                st.error("Please enter some text to analyze. Empty or whitespace-only input cannot be processed.")
            else:
                try:
                    from src.predict import predict_news
                    result = predict_news(news_text, model=pipeline)

                    st.markdown("---")
                    st.subheader("Analysis Findings")

                    # Classification Banner
                    if result.predicted_label == 1:
                        # Fake / Unreliable
                        st.error("### 🚩 Prediction: **Likely Fake**")
                        st.markdown(
                            "The linguistic features, vocabulary, and stylistic markers of this text exhibit characteristics "
                            "strongly resembling articles labeled as **Fake News** in the training dataset."
                        )
                    else:
                        # Real / Reliable
                        st.success("### 🟢 Prediction: **Likely Real**")
                        st.markdown(
                            "The linguistic style, tone, and vocabulary of this text align with patterns "
                            "typical of articles labeled as **Real News** in the training dataset."
                        )

                    # Probabilities Breakdown
                    if result.probabilities:
                        st.markdown("#### Estimated Resemblance Probabilities")
                        p_col1, p_col2 = st.columns(2)
                        real_p = result.probabilities.get("Real", 0.0)
                        fake_p = result.probabilities.get("Fake", 0.0)

                        with p_col1:
                            st.metric(
                                label="Likely Real Resemblance Score",
                                value=f"{real_p * 100:.1f}%",
                            )
                            st.progress(real_p)

                        with p_col2:
                            st.metric(
                                label="Likely Fake Resemblance Score",
                                value=f"{fake_p * 100:.1f}%",
                            )
                            st.progress(fake_p)

                        st.caption(
                            "Note: Probability estimates reflect logistic regression sigmoid outputs "
                            "relative to training distributions. They represent text-similarity scores, not absolute probabilities of truth."
                        )

                    # Context and Ethics Notice
                    st.warning(
                        f"⚠️ **Important Verification Notice:**\n\n{result.disclaimer}\n\n"
                        "**Crucial Reminder:** This prediction is strictly an automated stylistic classification based on patterns "
                        "learned from the dataset. It cannot verify real-world facts or confirm whether an event actually happened. "
                        "Always consult multiple verified, independent sources for critical news."
                    )

                except Exception as err:
                    st.error(f"Prediction failed: {err}")


# -----------------------------------------------------------------------------
# PAGE C: MODEL INFORMATION
# -----------------------------------------------------------------------------
elif page == "📊 Model Information":
    st.title("📊 Model Information & Evaluation Metrics")
    st.markdown("Detailed breakdown of model architecture, training configuration, and performance.")

    st.markdown("""
    ### 🔬 Technical Architecture
    - **Feature Extraction:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency)
      - Word unigrams and bigrams (`ngram_range=(1, 2)`)
      - English stop-word filtering
      - Sublinear term frequency dampening (`sublinear_tf=True`)
    - **Classification Algorithm:** `LogisticRegression`
      - Regularization: L2 Ridge penalty
      - Optimization: L-BFGS solver
      - Balanced class weights
    """)

    st.markdown("---")
    st.subheader("📈 Holdout Test Set Performance")

    metrics_data = get_latest_metrics()
    if metrics_data is None:
        st.info(
            "ℹ️ **No verified evaluation metrics found on disk.**\n\n"
            "Real evaluation metrics will be generated and saved to `reports/metrics_summary.json` "
            "as soon as you execute `python -m src.train_model`."
        )
    else:
        m = metrics_data.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{m.get('accuracy', 0.0) * 100:.2f}%")
        col2.metric("Precision (Fake)", f"{m.get('precision_fake', 0.0) * 100:.2f}%")
        col3.metric("Recall (Fake)", f"{m.get('recall_fake', 0.0) * 100:.2f}%")
        col4.metric("F1-Score (Fake)", f"{m.get('f1_score_fake', 0.0) * 100:.2f}%")

        st.caption(f"Evaluation Timestamp (UTC): {metrics_data.get('evaluation_timestamp', 'N/A')}")
        st.caption(f"Total Test Samples: {metrics_data.get('total_test_samples', 0):,}")

        # Confusion Matrix Image
        if CONFUSION_MATRIX_PATH.exists():
            st.markdown("### 🔲 Holdout Confusion Matrix")
            st.image(
                str(CONFUSION_MATRIX_PATH),
                caption="Confusion Matrix on Independent Stratified Test Set",
                use_container_width=True,
            )

        # Raw Metrics Inspector
        with st.expander("View Full Evaluation JSON Report"):
            st.json(metrics_data)


# -----------------------------------------------------------------------------
# PAGE D: ABOUT PROJECT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 🎯 Problem Statement
    The proliferation of digital misinformation, sensationalized content, and disinformation campaigns 
    presents a significant challenge to modern public discourse. Automated Natural Language Processing (NLP) 
    techniques can assist by highlighting linguistic markers, hyperbole, and structural cues commonly observed 
    in misleading articles.

    ### 🛠️ Technology Stack
    - **Programming Language:** Python 3.12
    - **Data Processing:** pandas, NumPy
    - **Machine Learning & NLP:** scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`)
    - **Web Framework:** Streamlit
    - **Visualizations:** Matplotlib
    - **Testing Framework:** pytest
    - **Deployment Platform:** Streamlit Community Cloud

    ### ⚠️ Known Limitations
    1. **Domain & Temporal Shift:** A model trained on political articles from past election cycles may not generalize well to emerging topics (e.g., scientific health studies, technology releases, sports).
    2. **Satire & Parody:** Satirical publications (e.g., *The Onion*) use hyperbole for humor rather than deception, which style-based models frequently misclassify.
    3. **Sophisticated Misinformation:** A false narrative written in formal, dry journalistic prose will share stylistic features with legitimate news and may bypass linear text classifiers.
    4. **Short Text Limitations:** Single short sentences or tweets may lack sufficient vocabulary for confident TF-IDF representation.

    ### 🔮 Future Enhancements
    - Integration of entity-linking and claim-verification knowledge graphs.
    - Multi-domain training datasets covering science, entertainment, and financial reporting.
    - Explainability tools (e.g., SHAP / LIME or top informative TF-IDF feature weights display).
    - Confidence calibration techniques (e.g., Platt scaling or Isotonic regression).
    """)
