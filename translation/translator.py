"""
Translation Module — Fully Offline Russian → English
Uses argostranslate as the primary backend.
Falls back gracefully when the model is unavailable.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Unicode range for Cyrillic characters
_CYRILLIC_START = 0x0400
_CYRILLIC_END = 0x04FF
_RUSSIAN_RATIO_THRESHOLD = 0.35  # 35 % Cyrillic → treat as Russian


# ---------------------------------------------------------------------------
# Dependency / model availability checks
# ---------------------------------------------------------------------------

def check_argos_available() -> bool:
    """Return True if the argostranslate package can be imported."""
    try:
        import argostranslate.translate  # noqa: F401
        return True
    except ImportError:
        return False


def check_ru_en_model() -> bool:
    """
    Return True if the Russian → English Argos model is installed.
    """
    if not check_argos_available():
        return False
    try:
        import argostranslate.translate

        langs = argostranslate.translate.get_installed_languages()
        ru = next((lang for lang in langs if lang.code == "ru"), None)
        if ru is None:
            return False
        return any(t.to_lang.code == "en" for t in ru.translations_to)
    except Exception as exc:
        logger.warning(f"Could not verify Argos RU→EN model: {exc}")
        return False


def get_install_instructions() -> str:
    """Return step-by-step instructions for installing the RU→EN model."""
    return (
        "Install the Russian → English Argos Translate model:\n\n"
        "  Option A (automatic, requires internet once):\n"
        "    python -c \"\n"
        "import argostranslate.package\n"
        "argostranslate.package.update_package_index()\n"
        "pkgs = argostranslate.package.get_available_packages()\n"
        "pkg  = next(p for p in pkgs if p.from_code=='ru' and p.to_code=='en')\n"
        "argostranslate.package.install_from_path(pkg.download())\n"
        "    \"\n\n"
        "  Option B (manual):\n"
        "    1. Download translate-ru_en-*.argosmodel from\n"
        "       https://www.argosopentech.com/argospm/index/\n"
        "    2. argos-translate-cli --install translate-ru_en-*.argosmodel\n"
    )


def try_install_ru_en_model() -> bool:
    """
    Attempt to download and install the RU→EN model from the Argos index.
    Returns True on success.
    """
    if not check_argos_available():
        return False
    try:
        import argostranslate.package

        logger.info("Updating Argos package index …")
        argostranslate.package.update_package_index()

        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == "ru" and p.to_code == "en"),
            None,
        )
        if pkg is None:
            logger.error("RU→EN package not found in the Argos index.")
            return False

        logger.info("Downloading RU→EN model …")
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
        logger.info("RU→EN model installed successfully.")
        return True
    except Exception as exc:
        logger.error(f"Auto-install failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Language detection (lightweight, no external library required)
# ---------------------------------------------------------------------------

def is_russian(text: str) -> bool:
    """
    Heuristic: True when > 35 % of alphabetic characters are Cyrillic.
    """
    if not text.strip():
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    cyrillic = sum(
        1 for c in alpha_chars
        if _CYRILLIC_START <= ord(c) <= _CYRILLIC_END
    )
    return (cyrillic / len(alpha_chars)) >= _RUSSIAN_RATIO_THRESHOLD


def detect_language(text: str) -> str:
    """
    Simple rule-based language detector.
    Returns an ISO 639-1 code: 'ru' or 'en' (default).
    Extend this function for additional language support.
    """
    if is_russian(text):
        return "ru"
    return "en"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def _get_argos_translation():
    """
    Return the cached Argos Translation object for ru→en.
    Returns None if unavailable.
    """
    try:
        import argostranslate.translate

        langs = argostranslate.translate.get_installed_languages()
        ru_lang = next((l for l in langs if l.code == "ru"), None)
        if ru_lang is None:
            return None
        en_lang = next((l for l in langs if l.code == "en"), None)
        if en_lang is None:
            return None
        return ru_lang.get_translation(en_lang)
    except Exception as exc:
        logger.warning(f"Could not load Argos ru→en model: {exc}")
        return None


def translate_ru_to_en(text: str) -> str:
    """
    Translate a Russian string to English using Argos Translate.
    Returns the original text unchanged if translation is unavailable.
    """
    translation = _get_argos_translation()
    if translation is None:
        logger.warning("RU→EN model unavailable — returning original text.")
        return text
    try:
        result = translation.translate(text)
        return result
    except Exception as exc:
        logger.error(f"Translation error: {exc}")
        return text


def translate_text(text: str, enabled: bool = True) -> Tuple[str, bool]:
    """
    Translate *text* if it is detected as Russian and translation is enabled.

    Returns
    -------
    (translated_text, was_translated)
    """
    if not enabled or not text.strip():
        return text, False
    if is_russian(text):
        translated = translate_ru_to_en(text)
        return translated, translated != text
    return text, False


def translate_blocks(
    blocks: List[Dict[str, Any]],
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Translate every block's 'text' field in-place (or copy).

    Each block gains two new fields:
        translated_text : str   — translated (or original) text
        was_translated  : bool  — True when a translation was applied
    """
    for block in blocks:
        original = block.get("text", "")
        translated, changed = translate_text(original, enabled=enabled)
        block["translated_text"] = translated
        block["was_translated"] = changed
    return blocks
