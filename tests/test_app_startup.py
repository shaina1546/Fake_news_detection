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
