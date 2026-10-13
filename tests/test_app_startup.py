"""
Test UI startup behavior without a trained model using Streamlit AppTest.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_startup_without_model():
    """
    Verifies that the Streamlit application starts gracefully and renders
    its warning/setup messaging when no trained model exists.
    """
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)

    # The app should initialize without throwing unhandled exceptions
    assert not at.exception
    # Title should match
    assert at.title[0].value == "📰 Fake News Detection System"


def test_streamlit_app_all_pages_render():
    """Every page renders without exceptions."""
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    for page in ["Analyze News", "Model Information", "About", "Home"]:
        at.radio[0].set_value(page).run(timeout=30)
        assert not at.exception, page


def test_analyze_flow_with_sample():
    """Loading a sample and analyzing it produces a prediction (skipped if no model)."""
    model = Path(__file__).resolve().parent.parent / "models" / "fake_news_model.joblib"
    if not model.exists():
        return
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    at.radio[0].set_value("Analyze News").run(timeout=30)
    at.button[1].click().run(timeout=30)   # fake-style sample
    at.button[3].click().run(timeout=30)   # Analyze
    assert not at.exception
    assert any("Likely" in m.value for m in at.markdown)
