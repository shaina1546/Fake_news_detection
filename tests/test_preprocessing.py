"""
Unit tests for text_preprocessor module.
"""

from src.text_preprocessor import clean_text, preprocess_corpus


def test_clean_text_normal_string():
    raw = "  Breaking   News: Global climate summit begins today in Geneva! \n\t  "
    expected = "breaking news: global climate summit begins today in geneva!"
    assert clean_text(raw) == expected


def test_clean_text_empty_and_whitespace():
    assert clean_text("") == ""
    assert clean_text("    ") == ""
    assert clean_text("\n\t\r") == ""


def test_clean_text_none_handling():
    assert clean_text(None) == ""


def test_clean_text_non_string_types():
    assert clean_text(12345) == "12345"
    assert clean_text(45.67) == "45.67"


def test_clean_text_preserve_casing_flag():
    raw = "  IMPORTANT ANNOUNCEMENT!  "
    assert clean_text(raw, lowercase=False) == "IMPORTANT ANNOUNCEMENT!"


def test_preprocess_corpus():
    inputs = [
        "First article headline.",
        None,
        "   Another   article with    spaces   ",
        "",
    ]
    outputs = preprocess_corpus(inputs)
    assert len(outputs) == 4
    assert outputs[0] == "first article headline."
    assert outputs[1] == ""
    assert outputs[2] == "another article with spaces"
    assert outputs[3] == ""
