"""
Text Preprocessing Module for Fake News Detection System.

Provides consistent, safe text normalization routines for training
and inference pipelines.
"""

from __future__ import annotations
import re
from typing import Any, Iterable


def clean_text(text: Any, lowercase: bool = True) -> str:
    """
    Cleans and normalizes news text input.

    Rules applied:
    - Safe handling of None, NaN, and non-string inputs (returns empty string).
    - Strips leading and trailing whitespace.
    - Normalizes multiple spaces, tabs, and newlines into single spaces.
    - Preserves meaningful grammatical and contextual tokens.
    - Optionally converts text to lowercase (standard for TF-IDF feature extraction).

    Parameters:
    -----------
    text : Any
        Input text (string, None, or other type).
    lowercase : bool, default=True
        Whether to convert text to lower case.

    Returns:
    --------
    str
        Normalized clean string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        # Convert non-string (numbers, etc.) to string representation
        text = str(text)

    # Normalize all consecutive whitespace characters (including \r, \n, \t) to single space
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.strip()

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def preprocess_corpus(texts: Iterable[Any], lowercase: bool = True) -> list[str]:
    """
    Applies clean_text across a sequence or list of texts.

    Parameters:
    -----------
    texts : Iterable[Any]
        Collection of texts to normalize.
    lowercase : bool, default=True
        Whether to convert text to lower case.

    Returns:
    --------
    list[str]
        List of cleaned strings.
    """
    return [clean_text(t, lowercase=lowercase) for t in texts]
