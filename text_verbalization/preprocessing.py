"""
Preprocessing functions for RDF Knowledge Graph matching.
This module contains functions for tokenization, stop-word removal, and lowercasing
that can be used for both BM25 and TF-IDF candidate generation approaches.
"""

import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Configure logging
logger = logging.getLogger(__name__)

# Set up stop words
stop_words = set(stopwords.words("english"))
stop_words.add("type")
punctuation = set(string.punctuation)


def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stop words, and lowercasing.

    Args:
        text (str): The text to preprocess

    Returns:
        list: List of preprocessed tokens
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        word for word in tokens if word not in stop_words and word not in punctuation
    ]
    return filtered_tokens


def preprocess_corpus(corpus):
    """
    Preprocess a corpus of texts.

    Args:
        corpus (list): List of texts to preprocess

    Returns:
        list: List of preprocessed token lists
    """
    return [preprocess_text(text) for text in corpus]
