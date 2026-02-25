"""
PDF Reconstruction Module
Rebuilds a translated document as a PDF using ReportLab.

Layout strategy
---------------
1.  The original rasterised page is placed as a full-page background image.
2.  For each OCR text block, a white filled rectangle is drawn over the
    original text area, then the translated (or original) text is placed
    inside that rectangle using a proportional font size.
3.  Extracted sub-images from the original PDF are re-inserted at their
    original coordinates.

Coordinate system notes
-----------------------
• pytesseract returns pixel coordinates with origin at the **top-left**.
• ReportLab uses PDF user-space with origin at the **bottom-left**.
• When the page image is placed at its natural pixel size (1 px = 1 pt),
  the y-axis flip is:  pdf_y = page_height_px - (img_top_y + img_height)
"""

import io
import logging
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def _find_unicode_font() -> str:
    """
    Try to register a Unicode-capable TTF font for ReportLab.
    Falls back to 'Helvetica' (Latin-only) if none is found.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # Common paths for DejaVu / Liberation fonts
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            try:
                pdfmetrics.registerFont(TTFont("UniFont", path))
                return "UniFont"
            except Exception:
                continue
    except Exception:
        pass
    return "Helvetica"


def build_pdf(
    pages_data: List[Dict[str, Any]],
    page_images: List[PILImage.Image],
    include_background: bool = True,
    background_opacity: float = 1.0,
) -> bytes:
    """
    Build a translated PDF from OCR results and original page images.

    Parameters
    ----------
    pages_data         : List of per-page dicts produced by process_page().
                         Each dict must contain:
                             'width'  : page width in pixels
                             'height' : page height in pixels
                             'blocks' : list of OCR block dicts
                             'images' : list of embedded image dicts
    page_images        : Original rasterised page images (PIL).
    include_background : When True, each page starts with the original image
                         as background (recommended for layout fidelity).
    background_opacity : Unused future parameter (reserved).

    Returns
    -------
    Raw bytes of the generated PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError(
            "reportlab is required: pip install reportlab"
        ) from exc

    font_name = _find_unicode_font()
    output = io.BytesIO()

    c = canvas.Canvas(output)

    for page_idx, (page_data, page_img) in enumerate(
        zip(pages_data, page_images)
    ):
        page_w = page_data.get("width", page_img.width)
        page_h = page_data.get("height", page_img.height)

        c.setPageSize((page_w, page_h))

        # ── 1. Background image ──────────────────────────────────────────────
        if include_background and page_img is not None:
            try:
                buf = io.BytesIO()
                page_img.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                c.drawImage(
                    ImageReader(buf),
                    0, 0,
                    width=page_w,
                    height=page_h,
                    preserveAspectRatio=False,
                )
            except Exception as exc:
                logger.warning(f"Could not draw background image: {exc}")

        # ── 2. Re-insert embedded sub-images ────────────────────────────────
        for img_info in page_data.get("images", []):
            try:
                pil_img: Optional[PILImage.Image] = img_info.get("image")
                if pil_img is None:
                    continue
                # bbox from extract_embedded_images is in PDF points (top-left origin)
                bx, by, bw, bh = img_info["bbox"]
                pdf_y = page_h - by - bh  # flip y to bottom-left origin

                buf = io.BytesIO()
                pil_img.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                c.drawImage(
                    ImageReader(buf),
                    bx, pdf_y,
                    width=bw,
                    height=bh,
                    preserveAspectRatio=False,
                )
            except Exception as exc:
                logger.warning(f"Skipping embedded image: {exc}")

        # ── 3. Text blocks ───────────────────────────────────────────────────
        for block in page_data.get("blocks", []):
            text: str = block.get("translated_text", block.get("text", ""))
            if not text.strip():
                continue

            bx, by, bw, bh = block["bbox"]
            # Flip y-axis: OCR top-left → PDF bottom-left
            pdf_y = page_h - by - bh

            # Estimate font size from block height and line count
            words = block.get("words", [])
            if words:
                unique_tops = {round(w["top"] / 5) * 5 for w in words}
                num_lines = max(1, len(unique_tops))
            else:
                num_lines = max(1, text.count("\n") + 1)

            line_height_px = bh / num_lines
            font_size = max(6.0, min(24.0, line_height_px * 0.72))

            # White background rectangle to erase original text
            c.setFillColor(colors.white)
            c.rect(bx, pdf_y, bw, bh, fill=1, stroke=0)

            # Draw translated text
            c.setFillColor(colors.black)
            try:
                c.setFont(font_name, font_size)
            except Exception:
                c.setFont("Helvetica", font_size)

            # Split into lines and draw with word-wrap
            lines = text.split("\n") if "\n" in text else [text]
            text_obj = c.beginText(bx + 2, pdf_y + bh - font_size - 2)
            for line in lines:
                # Crude word-wrap: truncate at block width
                while c.stringWidth(line, font_name, font_size) > bw - 4 and len(line) > 1:
                    line = line[: len(line) - 1]
                text_obj.textLine(line)

            try:
                c.drawText(text_obj)
            except Exception as exc:
                logger.warning(f"Could not draw text block: {exc}")

        c.showPage()

    c.save()
    output.seek(0)
    return output.read()
