"""
PDF Reconstruction Module
=========================
Rebuilds a translated document as a PDF using ReportLab.

Layout strategy
---------------
1.  The original rasterised page is placed as a full-page background image,
    preserving photos, decorative elements and overall document feel.
2.  For each OCR text block a *semi-transparent* white rectangle is drawn
    over the original text area so the text is legible without completely
    erasing the background.
3.  Translated (or original) text is placed inside each rectangle using
    proper word-wrapping and automatic font-size fitting.
4.  Embedded sub-images are NOT re-drawn — they are already present in the
    background page image and re-drawing them causes double/misaligned copies.

Coordinate system
-----------------
pytesseract  →  origin TOP-LEFT,    y increases downward  (pixel space)
ReportLab    →  origin BOTTOM-LEFT, y increases upward    (point space)

With page size set to the image pixel dimensions (1 px = 1 pt):
    pdf_y = page_height - (ocr_y + ocr_height)
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Alpha for the white overlay behind translated text (0 = invisible, 1 = solid).
# Must be 1.0: any transparency lets the rasterised background text bleed
# through, creating a ghost / double-text effect on top of the overlay.
_TEXT_BOX_ALPHA = 1.0
_PADDING = 3          # px padding inside each text box
_MIN_FONT  = 6.0
_MAX_FONT  = 48.0
_LINE_LEADING = 1.25  # line-height factor (leading = font_size * factor)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _find_unicode_font() -> str:
    """
    Register a Unicode-capable TTF font with ReportLab and return its name.
    Checks a wide range of paths covering Windows, macOS and common Linux
    distributions.  Falls back to the built-in 'Helvetica' (Latin-only).
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\times.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux (DejaVu / Liberation)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            pdfmetrics.registerFont(TTFont("AppFont", path))
            logger.debug("PDF font: AppFont → %s", path)
            return "AppFont"
        except Exception:
            continue
    return "Helvetica"


# ---------------------------------------------------------------------------
# Word-wrap helpers
# ---------------------------------------------------------------------------

def _word_wrap(
    text: str,
    canvas: Any,
    font_name: str,
    font_size: float,
    max_width: float,
) -> List[str]:
    """
    Break *text* into lines that each fit within *max_width* using proper
    word-boundary wrapping.  Does NOT truncate words; a single very long
    word is placed on its own line even if it overflows slightly.
    """
    wrapped: List[str] = []
    for paragraph in (text or "").split("\n"):
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if canvas.stringWidth(candidate, font_name, font_size) <= max_width:
                current.append(word)
            else:
                if current:
                    wrapped.append(" ".join(current))
                # If a single word is already too wide, still put it alone
                current = [word]
        if current:
            wrapped.append(" ".join(current))
    return wrapped or [""]


def _fit_text(
    text: str,
    canvas: Any,
    font_name: str,
    box_w: float,
    box_h: float,
    start_size: float,
) -> Tuple[float, List[str]]:
    """
    Find the largest font size ≤ start_size at which *text* fully fits in
    the box (width × height).  Returns (font_size, wrapped_lines).

    If even _MIN_FONT doesn't fit vertically the minimum size is used anyway
    so that at least the first few lines are visible.
    """
    usable_w = max(box_w - _PADDING * 2, 10.0)
    size = min(start_size, _MAX_FONT)

    while size >= _MIN_FONT:
        lines = _word_wrap(text, canvas, font_name, size, usable_w)
        total_h = len(lines) * size * _LINE_LEADING
        if total_h <= box_h - _PADDING * 2:
            return size, lines
        size -= 0.5

    # Give up shrinking — return minimum size with wrapped lines
    lines = _word_wrap(text, canvas, font_name, _MIN_FONT, usable_w)
    return _MIN_FONT, lines


# ---------------------------------------------------------------------------
# Font-size estimation from OCR word metadata
# ---------------------------------------------------------------------------

def _estimate_font_size(block: Dict[str, Any], box_h: float) -> float:
    """
    Estimate a starting font size from the OCR word bounding boxes.
    Falls back to a fraction of the box height when word data is absent.
    """
    words = block.get("words", [])
    if words:
        # Average word height is a good proxy for cap-height ≈ font-size * 0.7
        avg_h = sum(w.get("height", 0) for w in words) / len(words)
        if avg_h > 0:
            return max(_MIN_FONT, min(_MAX_FONT, avg_h * 0.85))

    unique_tops = {round(w["top"] / 4) * 4 for w in words} if words else set()
    num_lines = max(1, len(unique_tops))
    return max(_MIN_FONT, min(_MAX_FONT, (box_h / num_lines) * 0.75))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_pdf(
    pages_data: List[Dict[str, Any]],
    page_images: List[PILImage.Image],
    include_background: bool = True,
) -> bytes:
    """
    Build a translated PDF from OCR results and original page images.

    Parameters
    ----------
    pages_data        : Per-page dicts from process_page().
    page_images       : Original rasterised page images (PIL), one per page.
    include_background: Draw each original page as a full-bleed background.

    Returns
    -------
    Raw bytes of the generated PDF.
    """
    try:
        from reportlab.lib.colors import Color
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError as exc:
        raise ImportError("reportlab is required: pip install reportlab") from exc

    font_name = _find_unicode_font()
    output = io.BytesIO()
    c = rl_canvas.Canvas(output)

    for page_data, page_img in zip(pages_data, page_images):
        page_w = float(page_data.get("width",  page_img.width))
        page_h = float(page_data.get("height", page_img.height))

        c.setPageSize((page_w, page_h))

        # ── 1. Background: original page image ──────────────────────────────
        if include_background and page_img is not None:
            try:
                buf = io.BytesIO()
                page_img.convert("RGB").save(buf, format="JPEG", quality=85)
                buf.seek(0)
                c.drawImage(
                    ImageReader(buf),
                    0, 0, width=page_w, height=page_h,
                    preserveAspectRatio=False,
                )
            except Exception as exc:
                logger.warning("Could not draw background: %s", exc)

        # ── 2. Text blocks ───────────────────────────────────────────────────
        #    NOTE: embedded images are intentionally NOT re-drawn here.
        #    They already appear in the background page image above.
        for block in page_data.get("blocks", []):
            text: str = block.get("translated_text", block.get("text", ""))
            if not text.strip():
                continue

            bx, by, bw, bh = (float(v) for v in block["bbox"])
            if bw < 4 or bh < 4:
                continue

            # Flip y-axis: OCR top-left origin → PDF bottom-left origin
            pdf_y = page_h - by - bh

            # Estimate starting font size from word metadata
            start_size = _estimate_font_size(block, bh)

            # Find largest font size where text fits the box
            try:
                c.setFont(font_name, start_size)
            except Exception:
                font_name = "Helvetica"
                c.setFont(font_name, start_size)

            font_size, lines = _fit_text(text, c, font_name, bw, bh, start_size)

            # ── Semi-transparent white backdrop ─────────────────────────────
            c.saveState()
            c.setFillColor(Color(1, 1, 1, alpha=_TEXT_BOX_ALPHA))
            c.rect(bx, pdf_y, bw, bh, fill=1, stroke=0)
            c.restoreState()

            # ── Draw text ────────────────────────────────────────────────────
            c.saveState()
            try:
                c.setFont(font_name, font_size)
            except Exception:
                c.setFont("Helvetica", font_size)
            c.setFillColor(Color(0, 0, 0, alpha=1))

            leading = font_size * _LINE_LEADING
            # Start baseline just inside the top of the box
            x_text = bx + _PADDING
            y_text = pdf_y + bh - font_size - _PADDING

            for line in lines:
                if y_text < pdf_y:          # stop if we've left the box
                    break
                try:
                    c.drawString(x_text, y_text, line)
                except Exception as exc:
                    logger.debug("drawString failed: %s", exc)
                y_text -= leading

            c.restoreState()

        c.showPage()

    c.save()
    output.seek(0)
    return output.read()
