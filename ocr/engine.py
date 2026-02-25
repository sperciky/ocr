"""
OCR Engine Module
Wraps pytesseract to extract text with full bounding-box metadata.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Tesseract Page Segmentation Modes (PSM)
PSM_AUTO = 3
PSM_SINGLE_BLOCK = 6


def check_tesseract() -> Tuple[bool, str]:
    """
    Verify Tesseract is installed and return its version string.

    Returns
    -------
    (is_available, version_or_error_message)
    """
    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except pytesseract.TesseractNotFoundError:
        return False, (
            "Tesseract not found in PATH.\n"
            "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS   : brew install tesseract tesseract-lang\n"
            "  Linux   : sudo apt-get install tesseract-ocr tesseract-ocr-rus"
        )
    except Exception as exc:
        return False, f"Tesseract error: {exc}"


def get_available_languages() -> List[str]:
    """Return sorted list of installed Tesseract language packs."""
    try:
        langs = pytesseract.get_languages(config="")
        return sorted(langs)
    except Exception:
        return ["eng", "rus"]


def run_ocr(
    image: Image.Image,
    lang: str = "eng+rus",
    confidence_threshold: float = 30.0,
    psm: int = PSM_AUTO,
) -> pd.DataFrame:
    """
    Run Tesseract OCR and return a structured DataFrame with word-level metadata.

    Columns returned:
        level, page_num, block_num, par_num, line_num, word_num,
        left, top, width, height, conf, text

    Parameters
    ----------
    image:                PIL Image to process.
    lang:                 Tesseract language string (e.g. 'eng+rus').
    confidence_threshold: Drop words with conf < this value (0 = keep all).
    psm:                  Tesseract Page Segmentation Mode.
    """
    try:
        config = f"--psm {psm}"
        df: pd.DataFrame = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DATAFRAME,
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found in PATH.\n"
            "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS   : brew install tesseract tesseract-lang\n"
            "  Linux   : sudo apt-get install tesseract-ocr tesseract-ocr-rus"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc

    # --- Clean up DataFrame ---
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    # Keep only word-level rows (level == 5) and container rows used for grouping
    df = df[df["conf"].notna()].copy()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1)

    # Apply confidence filter (conf == -1 means a non-leaf container row)
    word_rows = df[df["level"] == 5].copy()
    if confidence_threshold > 0:
        word_rows = word_rows[word_rows["conf"] >= confidence_threshold]

    return word_rows.reset_index(drop=True)


def ocr_to_blocks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Aggregate word-level OCR rows into logical text blocks.

    Each block groups all words that share the same `block_num` and
    computes the enclosing bounding box plus mean confidence.

    Returns
    -------
    List of dicts with keys:
        block_num, bbox [x, y, w, h], text, words, confidence, par_num
    """
    if df.empty:
        return []

    blocks: List[Dict[str, Any]] = []

    for block_num, group in df.groupby("block_num"):
        words = group.copy()
        if words.empty:
            continue

        text = " ".join(words["text"].tolist())
        if not text.strip():
            continue

        x1 = int(words["left"].min())
        y1 = int(words["top"].min())
        x2 = int((words["left"] + words["width"]).max())
        y2 = int((words["top"] + words["height"]).max())

        avg_conf = float(words["conf"].mean())
        par_num = int(words["par_num"].iloc[0]) if "par_num" in words else 0

        word_records = words[
            ["left", "top", "width", "height", "text", "conf"]
        ].to_dict("records")

        blocks.append(
            {
                "block_num": int(block_num),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "text": text,
                "words": word_records,
                "confidence": avg_conf,
                "par_num": par_num,
            }
        )

    # Sort top-to-bottom, left-to-right by default
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return blocks


def get_word_level_data(
    image: Image.Image,
    lang: str = "eng+rus",
    confidence_threshold: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: run OCR and return blocks directly.
    """
    df = run_ocr(image, lang=lang, confidence_threshold=confidence_threshold)
    return ocr_to_blocks(df)
