"""
This module provides utilities for other modules.
"""
import string
import re
import logging
import configparser
from numba import jit
import numpy as np


# Japanese and English punctuation patterns
punctuation_pattern = (
    rf"\[\u2150-\u218F\u2190-\u21FF\u2460-\u26FF\u3003-\u300B\u300E-\u301B"
    rf"\u301D-\u303F\r\n{re.escape(string.punctuation)}\]+$"
)

# Japanese stop words
ja_stop_words = [
    "あそこ",
    "あっ",
    "あの",
    "あのかた",
    "あの人",
    "あり",
    "あります",
    "ある",
    "あれ",
    "い",
    "いう",
    "います",
    "いる",
    "う",
    "うち",
    "え",
    "お",
    "および",
    "おり",
    "おります",
    "か",
    "かつて",
    "から",
    "が",
    "き",
    "ここ",
    "こちら",
    "こと",
    "この",
    "これ",
    "これら",
    "さ",
    "さらに",
    "し",
    "しかし",
    "する",
    "ず",
    "せ",
    "せる",
    "そこ",
    "そして",
    "その",
    "その他",
    "その後",
    "それ",
    "それぞれ",
    "それで",
    "た",
    "ただし",
    "たち",
    "ため",
    "たり",
    "だ",
    "だっ",
    "だれ",
    "つ",
    "て",
    "で",
    "でき",
    "できる",
    "です",
    "では",
    "でも",
    "と",
    "という",
    "といった",
    "とき",
    "ところ",
    "として",
    "とともに",
    "とも",
    "と共に",
    "どこ",
    "どの",
    "な",
    "ない",
    "なお",
    "なかっ",
    "ながら",
    "なく",
    "なっ",
    "など",
    "なに",
    "なら",
    "なり",
    "なる",
    "なん",
    "に",
    "において",
    "における",
    "について",
    "にて",
    "によって",
    "により",
    "による",
    "に対して",
    "に対する",
    "に関する",
    "の",
    "ので",
    "のみ",
    "は",
    "ば",
    "へ",
    "ほか",
    "ほとんど",
    "ほど",
    "ます",
    "また",
    "または",
    "まで",
    "も",
    "もの",
    "ものの",
    "や",
    "よう",
    "より",
    "ら",
    "られ",
    "られる",
    "れ",
    "れる",
    "を",
    "ん",
    "何",
    "及び",
    "彼",
    "彼女",
    "我々",
    "特に",
    "私",
    "私達",
    "貴方",
    "貴方方",
]


punctuation_pattern_compiled = re.compile(punctuation_pattern)


@jit(nopython=True)
def cosine_similarity_numba(u_vec: np.ndarray, v_vec: np.ndarray):
    """
    Compute cosine similarity between two vectors.

    Args:
        u_vec (np.ndarray): First vector.
        v_vec (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between u_vec and v_vec.
    """
    u_dot_v = 0
    u_norm_squared = 0
    v_norm_squared = 0

    for i in range(u_vec.shape[0]):
        u_dot_v += u_vec[i] * v_vec[i]
        u_norm_squared += u_vec[i] * u_vec[i]
        v_norm_squared += v_vec[i] * v_vec[i]

    cos_theta = 1

    if u_norm_squared != 0 and v_norm_squared != 0:
        cos_theta = u_dot_v / np.sqrt(u_norm_squared * v_norm_squared)

    return cos_theta


def get_config():
    """
    Used to get config parameters for inside the app

    Returns:
    ConfigParser: Returns config parser object
    """
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    return config

def get_mock_ocr_config():
    """
    Used to get mock ocr config parameters for simulating OCR service

    Returns:
    ConfigParser: Returns config parser object
    """
    config = configparser.ConfigParser()
    config.read("config/mock_ocr.ini")
    return config

def is_punctuation(data: str):
    """
    Check if the given token is a punctuation or not.

    Args:
    data (str): Token to check.

    Returns:
    bool: True if the token is a punctuation, False otherwise.
    """
    return data in string.punctuation or re.search(
        punctuation_pattern_compiled, data
    )


def remove_punctuation(data: str):
    """
    Remove punctuation from a string.

    Args:
    data (str): Input string.

    Returns:
    str: String without punctuation.
    """
    return re.sub(punctuation_pattern, "", data)


def remove_ja_stop_words(data: str):
    """
    Remove Japanese stop words from a string.

    Args:
    data (str): Input string.

    Returns:
    str: String without Japanese stop words.
    """
    return re.sub(ja_stop_words, "", data)


def setup_logging(log_file="logs/logfile.log"):
    """
    Setup logging configuration.

    Args:
    log_file (str): Name of the log file.
    """

    # Create a logger instance
    logger_instance = logging.getLogger(__name__)
    logger_instance.setLevel(logging.INFO)

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler and set level to debug
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Add formatter to console handler and file handler
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add console handler and file handler to logger
    logger_instance.addHandler(console_handler)
    logger_instance.addHandler(file_handler)

    return logger_instance


# Initialize logger
logger = setup_logging()
