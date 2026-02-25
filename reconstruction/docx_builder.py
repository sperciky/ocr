"""
DOCX Reconstruction Module
Builds a Word document from OCR results using python-docx.

Since DOCX does not support absolute pixel positioning in a portable way,
this builder uses a structured approach:
  • Each PDF page becomes a section.
  • A thumbnail of the original page is inserted.
  • Paragraphs follow in detected reading order.
  • Translated paragraphs are marked with a light-blue highlight.
  • A JSON metadata comment is appended at the end.
"""

import io
import json
import logging
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def _add_page_thumbnail(
    doc: Any,
    page_img: PILImage.Image,
    max_width_inches: float = 5.0,
) -> None:
    """Insert a thumbnail of the original page into the document."""
    try:
        from docx.shared import Inches

        buf = io.BytesIO()
        thumb = page_img.copy()
        # Limit thumbnail width while preserving aspect ratio
        ratio = max_width_inches * 96 / thumb.width  # 96 px/inch approx
        new_w = int(thumb.width * ratio)
        new_h = int(thumb.height * ratio)
        thumb = thumb.resize((new_w, new_h), PILImage.LANCZOS)
        thumb.save(buf, format="PNG")
        buf.seek(0)
        doc.add_picture(buf, width=Inches(max_width_inches))
    except Exception as exc:
        logger.warning(f"Could not insert page thumbnail: {exc}")


def _add_heading(doc: Any, text: str, level: int = 2) -> None:
    doc.add_heading(text, level=level)


def _add_text_block(
    doc: Any,
    original: str,
    translated: str,
    was_translated: bool,
    confidence: float,
) -> None:
    """Add a single OCR text block as a paragraph (with optional annotation)."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.shared import Pt, RGBColor

    if was_translated:
        # Show original in grey italic, then translated in normal black
        para = doc.add_paragraph()

        orig_run = para.add_run(f"[RU] {original}")
        orig_run.italic = True
        orig_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        orig_run.font.size = Pt(9)

        para.add_run("\n")

        trans_run = para.add_run(translated)
        trans_run.bold = False
        trans_run.font.size = Pt(11)
    else:
        para = doc.add_paragraph(translated)

    # Confidence as a small caption
    caption = doc.add_paragraph(f"  ↳ confidence: {confidence:.1f}%")
    caption.runs[0].font.size = Pt(8)
    caption.runs[0].font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)


def build_docx(
    pages_data: List[Dict[str, Any]],
    page_images: List[PILImage.Image],
    include_thumbnails: bool = True,
) -> bytes:
    """
    Build a DOCX document from OCR/translation results.

    Parameters
    ----------
    pages_data        : Per-page data dicts from process_page().
    page_images       : Original page images (PIL).
    include_thumbnails: Insert page preview thumbnails.

    Returns
    -------
    Raw bytes of the .docx file.
    """
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as exc:
        raise ImportError(
            "python-docx is required: pip install python-docx"
        ) from exc

    doc = Document()

    # Document title
    title = doc.add_heading("OCR Translation Result", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for page_idx, (page_data, page_img) in enumerate(
        zip(pages_data, page_images)
    ):
        page_num = page_data.get("page_number", page_idx + 1)

        if page_idx > 0:
            doc.add_page_break()

        _add_heading(doc, f"Page {page_num}", level=2)

        # Metadata line
        meta = (
            f"Words: {page_data.get('word_count', 0)}  |  "
            f"Translated: {page_data.get('translation_word_count', 0)}  |  "
            f"Columns: {page_data.get('num_columns', 1)}  |  "
            f"Time: {page_data.get('processing_time', 0):.2f}s"
        )
        meta_para = doc.add_paragraph(meta)
        meta_para.runs[0].font.size = Pt(9)
        meta_para.runs[0].italic = True

        # Page thumbnail
        if include_thumbnails and page_img is not None:
            _add_page_thumbnail(doc, page_img, max_width_inches=5.0)

        doc.add_paragraph()  # spacer

        _add_heading(doc, "Extracted & Translated Text", level=3)

        blocks = page_data.get("blocks", [])
        if not blocks:
            doc.add_paragraph("(No text detected on this page.)")
        else:
            for block in blocks:
                original = block.get("text", "")
                translated = block.get("translated_text", original)
                was_translated = block.get("was_translated", False)
                confidence = block.get("confidence", 0.0)

                if not original.strip():
                    continue

                _add_text_block(
                    doc,
                    original=original,
                    translated=translated,
                    was_translated=was_translated,
                    confidence=confidence,
                )

    # Append JSON export as a code block at the end
    doc.add_page_break()
    _add_heading(doc, "JSON Metadata Export", level=2)

    json_payload = {
        "pages": [
            {
                "page_number": p.get("page_number"),
                "blocks": [
                    {
                        "bbox": b.get("bbox"),
                        "original_text": b.get("text"),
                        "translated_text": b.get("translated_text"),
                        "confidence": round(b.get("confidence", 0), 2),
                    }
                    for b in p.get("blocks", [])
                ],
            }
            for p in pages_data
        ]
    }
    json_str = json.dumps(json_payload, ensure_ascii=False, indent=2)

    json_para = doc.add_paragraph(json_str)
    json_para.runs[0].font.name = "Courier New"
    json_para.runs[0].font.size = Pt(7)

    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return output.read()
