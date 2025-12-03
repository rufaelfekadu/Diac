import unicodedata
from pyarabic import araby


def remove_diacritics(text: str) -> str:
    return araby.strip_diacritics(text)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)
