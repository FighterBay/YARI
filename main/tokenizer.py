"""
This module provides tokenizer for Japanese(as of now).
"""

from janome.tokenizer import Tokenizer
from main import utils


def tokenize_text(text: str):
    """
    Tokenize the input text using Janome tokenizer.

     Args:
        text (str): Text to be tokenized

    Returns:
        list: A list of tokens
    """
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def process_tokens(tokens: list):
    """
    Process tokens by removing whitespace, punctuation, and stop words.

     Args:
        tokens (list): A list of tokens

    Returns:
        list: A list of processed tokens
    """
    processed_tokens = [
        token.surface
        for token in tokens
        if token.surface.strip()
        and not utils.is_punctuation(token.surface)
        and token.surface not in utils.ja_stop_words
    ]
    return processed_tokens


def tokenize_and_join_with_whitespace(text: str):
    """
    Tokenize the input text and join the processed tokens with whitespace.

    Args:
    text (str): The input text to be tokenized and processed.

    Returns:
    str: The processed text with tokens joined by whitespace.
    """
    tokens = tokenize_text(text)
    processed_tokens = process_tokens(tokens)
    return " ".join(processed_tokens)


def tokenize_and_construct_redis_query(text: str):
    """
    Tokenize the input text and return a list of processed tokens.

    Args:
    text (str): The input text to be tokenized and processed.

    Returns:
    list: A list of processed tokens.
    """
    tokens = tokenize_text(text)
    processed_tokens = process_tokens(tokens)
    return processed_tokens


def split_japanese_sentences(text: str):
    """
    Split the Japanese text into sentences.

    Args:
    text (str): The input Japanese text to be split into sentences.

    Returns:
    list: A list of sentences extracted from the input text.
    """
    tokens = tokenize_text(text)
    sentences = []
    current_sentence = []

    for token in tokens:
        if utils.is_punctuation(token.surface):
            if current_sentence:
                current_sentence.append(token.surface)
                sentence = "".join(current_sentence)
                sentences.append(sentence.strip())
                current_sentence = []
        else:
            current_sentence.append(token.surface)

    if current_sentence:
        sentence = "".join(current_sentence)
        sentences.append(sentence.strip())

    return sentences
