"""
OCR Recipe Translator — Main Streamlit Application
===================================================
Offline OCR (pytesseract) with Russian → English translation (Argos Translate).
Supports PDF and image inputs; outputs translated PDF, DOCX, or plain text.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OCR Document Translator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stAlert { border-radius: 8px; }
    .metric-label { font-size: 0.85rem !important; }
    div[data-testid="stExpander"] > div { padding: 0.5rem 1rem; }
    .translated-badge {
        background: #d4edda; color: #155724;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.78rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _check_all_dependencies() -> List[Dict[str, str]]:
    """
    Run once (cached) to detect missing or misconfigured dependencies.
    Returns a list of issue dicts: {name, error, fix}.
    """
    issues: List[Dict[str, str]] = []

    # Tesseract
    try:
        from ocr.engine import check_tesseract
        ok, msg = check_tesseract()
        if not ok:
            issues.append({"name": "Tesseract OCR", "error": msg, "fix": msg})
    except Exception as exc:
        issues.append({"name": "Tesseract OCR", "error": str(exc), "fix": str(exc)})

    # pdf2image / Poppler
    try:
        from pdf2image import convert_from_bytes  # noqa: F401
    except ImportError:
        issues.append(
            {
                "name": "pdf2image / Poppler",
                "error": "pdf2image is not installed.",
                "fix": (
                    "pip install pdf2image\n"
                    "Then install Poppler:\n"
                    "  macOS  : brew install poppler\n"
                    "  Linux  : sudo apt-get install poppler-utils\n"
                    "  Windows: https://github.com/oschwartz10612/poppler-windows"
                ),
            }
        )

    # Argos Translate package
    try:
        import argostranslate.translate  # noqa: F401
    except ImportError:
        issues.append(
            {
                "name": "argostranslate",
                "error": "argostranslate package is not installed.",
                "fix": "pip install argostranslate",
            }
        )
        return issues  # model check would fail anyway

    # Argos RU→EN model
    try:
        from translation.translator import check_ru_en_model, get_install_instructions
        if not check_ru_en_model():
            issues.append(
                {
                    "name": "Argos RU→EN Model",
                    "error": "The Russian → English translation model is not installed.",
                    "fix": get_install_instructions(),
                }
            )
    except Exception as exc:
        issues.append({"name": "Argos RU→EN Model", "error": str(exc), "fix": str(exc)})

    return issues


def render_dependency_warnings(issues: List[Dict[str, str]]) -> bool:
    """
    Render dependency issues in the UI.
    Returns True if Tesseract is missing (critical — app cannot proceed).
    """
    if not issues:
        return False

    tesseract_missing = any(i["name"] == "Tesseract OCR" for i in issues)
    translation_missing = any("RU→EN" in i["name"] for i in issues)

    if tesseract_missing:
        st.error("❌ **Tesseract OCR** is not installed. The application cannot run without it.")

    for issue in issues:
        severity = "error" if issue["name"] == "Tesseract OCR" else "warning"
        with st.expander(
            f"{'❌' if severity == 'error' else '⚠️'} {issue['name']}: {issue['error']}",
            expanded=tesseract_missing,
        ):
            st.code(issue["fix"], language="bash")

    if translation_missing and not tesseract_missing:
        st.warning(
            "⚠️ Translation model not found — OCR will still run but Russian text "
            "will **not** be translated. See the expander above for install steps."
        )

    return tesseract_missing


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def _draw_bounding_boxes(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
    show_confidence: bool = False,
) -> Image.Image:
    """
    Return a copy of *image* with coloured bounding boxes overlaid.
    Green = original text kept; Red = translated from Russian.
    """
    img_copy = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for block in blocks:
        x, y, w, h = block["bbox"]
        was_translated = block.get("was_translated", False)
        conf = block.get("confidence", 100)

        colour = (220, 53, 69, 120) if was_translated else (40, 167, 69, 100)
        outline = (220, 53, 69, 255) if was_translated else (40, 167, 69, 255)

        draw.rectangle([x, y, x + w, y + h], fill=colour, outline=outline, width=2)

        if show_confidence:
            draw.text((x + 2, y + 2), f"{conf:.0f}%", fill=(0, 0, 0, 200))

    return Image.alpha_composite(img_copy, overlay).convert("RGB")


def process_single_page(
    page_img: Image.Image,
    embedded_images: List[Dict[str, Any]],
    page_num: int,
    lang: str,
    confidence_threshold: float,
    preprocess_opts: Dict[str, bool],
    translate_enabled: bool,
) -> Dict[str, Any]:
    """
    Full processing pipeline for a single page:
    preprocess → OCR → layout → translate.
    """
    from ocr.engine import run_ocr, ocr_to_blocks
    from ocr.layout import detect_columns, sort_blocks_reading_order
    from ocr.preprocessing import preprocess_image, pil_to_cv2, cv2_to_pil
    from translation.translator import translate_blocks
    from utils.file_handler import get_page_dimensions

    t_start = time.perf_counter()

    # ── Preprocessing ──────────────────────────────────────────────────────
    cv2_img = pil_to_cv2(page_img)
    processed_cv2 = preprocess_image(
        cv2_img,
        grayscale=preprocess_opts.get("grayscale", False),
        threshold=preprocess_opts.get("threshold", False),
        denoise_img=preprocess_opts.get("denoise", False),
        deskew_img=preprocess_opts.get("deskew", False),
    )
    processed_pil = cv2_to_pil(processed_cv2)

    # ── OCR ────────────────────────────────────────────────────────────────
    ocr_df = run_ocr(
        processed_pil,
        lang=lang,
        confidence_threshold=confidence_threshold,
    )
    blocks = ocr_to_blocks(ocr_df)

    # ── Layout ─────────────────────────────────────────────────────────────
    page_w, page_h = get_page_dimensions(page_img)
    num_cols = detect_columns(blocks, page_w)
    blocks = sort_blocks_reading_order(blocks, num_cols, page_w)

    # ── Translation ────────────────────────────────────────────────────────
    blocks = translate_blocks(blocks, enabled=translate_enabled)

    # Filter embedded images for this page
    page_embedded = [img for img in embedded_images if img.get("page") == page_num]

    elapsed = time.perf_counter() - t_start

    return {
        "page_number": page_num + 1,
        "width": page_w,
        "height": page_h,
        "blocks": blocks,
        "images": page_embedded,
        "num_columns": num_cols,
        "processing_time": elapsed,
        "word_count": sum(len(b["text"].split()) for b in blocks),
        "translation_word_count": sum(
            len(b.get("translated_text", "").split())
            for b in blocks
            if b.get("was_translated")
        ),
    }


def build_json_export(
    filename: str,
    pages_data: List[Dict[str, Any]],
) -> bytes:
    """Serialise OCR results to the canonical JSON format."""
    payload = {
        "filename": filename,
        "pages": [
            {
                "page_number": p["page_number"],
                "blocks": [
                    {
                        "bbox": b["bbox"],
                        "original_text": b.get("text", ""),
                        "translated_text": b.get("translated_text", b.get("text", "")),
                        "confidence": round(b.get("confidence", 0.0), 2),
                    }
                    for b in p["blocks"]
                ],
            }
            for p in pages_data
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def build_plain_text_export(pages_data: List[Dict[str, Any]]) -> bytes:
    """Produce a plain-text document with translated content."""
    lines: List[str] = []
    for page in pages_data:
        lines.append(f"{'=' * 60}")
        lines.append(f"  Page {page['page_number']}")
        lines.append(f"{'=' * 60}")
        for block in page["blocks"]:
            text = block.get("translated_text", block.get("text", ""))
            if text.strip():
                lines.append(text)
                lines.append("")
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return current settings."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/"
            "Single_line_diagram_of_Tesseract_OCR_engine.svg/320px-"
            "Single_line_diagram_of_Tesseract_OCR_engine.svg.png",
            use_column_width=True,
        ) if False else None  # placeholder — no external URLs used

        st.title("⚙️ Settings")
        st.markdown("---")

        # ── OCR ───────────────────────────────────────────────────────────
        st.subheader("🔍 OCR")
        lang_map = {
            "English + Russian (default)": "eng+rus",
            "English only": "eng",
            "Russian only": "rus",
            "English + German": "eng+deu",
            "English + French": "eng+fra",
            "English + Italian": "eng+ita",
            "English + Czech": "eng+ces",
            "All supported": "eng+rus+deu+fra+ita+ces",
        }
        selected_lang = st.selectbox(
            "OCR Language Pack",
            list(lang_map.keys()),
            index=0,
            help="Language(s) passed to Tesseract. Use '+' combinations for multi-language docs.",
        )
        ocr_lang = lang_map[selected_lang]

        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="Discard OCR words below this confidence. Lower = more text, possibly noisy.",
        )

        dpi = st.select_slider(
            "Rendering DPI (PDF only)",
            options=[100, 150, 200, 250, 300],
            value=200,
            help="Higher DPI → better OCR accuracy but slower processing and more memory.",
        )

        st.markdown("---")

        # ── Preprocessing ─────────────────────────────────────────────────
        st.subheader("🖼️ Preprocessing")
        preprocess_opts = {
            "grayscale": st.checkbox("Grayscale", value=False, help="Convert to greyscale before OCR."),
            "threshold": st.checkbox(
                "Binarize (Otsu)",
                value=False,
                help="Apply Otsu thresholding. Disables grayscale option when enabled.",
            ),
            "denoise": st.checkbox("Denoise", value=False, help="Non-Local Means denoising."),
            "deskew": st.checkbox("Deskew", value=False, help="Auto-correct document skew angle."),
        }

        st.markdown("---")

        # ── Translation ───────────────────────────────────────────────────
        st.subheader("🌐 Translation")
        translate_enabled = st.checkbox(
            "Enable RU → EN Translation",
            value=True,
            help="Translate detected Russian text to English using Argos Translate (offline).",
        )

        st.markdown("---")

        # ── Display ───────────────────────────────────────────────────────
        st.subheader("🖥️ Display")
        show_bbox = st.checkbox(
            "Show Bounding Boxes",
            value=False,
            help="Overlay OCR bounding boxes on the page preview.\nGreen = original, Red = translated.",
        )
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display per-block OCR confidence in the text panel.",
        )

        st.markdown("---")

        # ── Output ────────────────────────────────────────────────────────
        st.subheader("📤 Output Format")
        output_format = st.radio(
            "Generate",
            ["Translated PDF", "Translated DOCX", "Raw Text Only"],
            index=0,
        )

    return {
        "ocr_lang": ocr_lang,
        "confidence_threshold": float(confidence_threshold),
        "dpi": int(dpi),
        "preprocess_opts": preprocess_opts,
        "translate_enabled": translate_enabled,
        "show_bbox": show_bbox,
        "show_confidence": show_confidence,
        "output_format": output_format,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def render_summary_metrics(pages_data: List[Dict[str, Any]]) -> None:
    total_pages = len(pages_data)
    total_words = sum(p["word_count"] for p in pages_data)
    total_translated = sum(p["translation_word_count"] for p in pages_data)
    total_time = sum(p["processing_time"] for p in pages_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pages", total_pages)
    c2.metric("Words Detected", f"{total_words:,}")
    c3.metric("Words Translated", f"{total_translated:,}")
    c4.metric("Total Time", f"{total_time:.1f} s")
    c5.metric(
        "Translation %",
        f"{(total_translated / total_words * 100):.0f}%" if total_words else "—",
    )


def render_page_results(
    page_data: Dict[str, Any],
    page_img: Image.Image,
    settings: Dict[str, Any],
) -> None:
    """Render a single page's results inside an expander."""
    page_num = page_data["page_number"]
    blocks = page_data["blocks"]

    with st.expander(
        f"📄 Page {page_num}  —  "
        f"{page_data['word_count']} words  |  "
        f"{page_data['num_columns']} col(s)  |  "
        f"{page_data['processing_time']:.2f}s",
        expanded=(page_num == 1),
    ):
        col_preview, col_text = st.columns([1, 1], gap="medium")

        # ── Left: page preview ─────────────────────────────────────────────
        with col_preview:
            st.markdown("**Original Page**")
            if settings["show_bbox"] and blocks:
                preview = _draw_bounding_boxes(
                    page_img,
                    blocks,
                    show_confidence=False,
                )
                st.image(preview, use_column_width=True)
                st.caption("🟢 Kept original  🔴 Translated from Russian")
            else:
                st.image(page_img, use_column_width=True)

            if page_data["num_columns"] > 1:
                st.info(f"📊 Multi-column layout detected ({page_data['num_columns']} columns)")

            st.caption(
                f"Page size: {page_data['width']} × {page_data['height']} px  |  "
                f"Embedded images: {len(page_data.get('images', []))}"
            )

        # ── Right: text blocks ─────────────────────────────────────────────
        with col_text:
            st.markdown("**Extracted & Translated Text**")

            if not blocks:
                st.info("No text detected on this page.")
            else:
                for block in blocks:
                    original = block.get("text", "")
                    translated = block.get("translated_text", original)
                    was_translated = block.get("was_translated", False)
                    conf = block.get("confidence", 0.0)

                    if not original.strip():
                        continue

                    with st.container():
                        if was_translated:
                            st.markdown(
                                '<span class="translated-badge">🌐 Translated</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<small style='color:grey'>**RU:** {original}</small>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**EN:** {translated}")
                        else:
                            st.write(translated)

                        if settings["show_confidence"]:
                            st.caption(f"Confidence: {conf:.1f}%")
                        st.divider()


def render_download_section(
    filename: str,
    pages_data: List[Dict[str, Any]],
    page_images: List[Image.Image],
    output_format: str,
) -> None:
    st.markdown("---")
    st.subheader("⬇️ Download Results")

    stem = Path(filename).stem
    col1, col2, col3 = st.columns(3)

    # ── JSON export ────────────────────────────────────────────────────────
    with col1:
        json_bytes = build_json_export(filename, pages_data)
        st.download_button(
            label="📊 JSON (OCR data)",
            data=json_bytes,
            file_name=f"{stem}_ocr.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Plain text ─────────────────────────────────────────────────────────
    with col2:
        text_bytes = build_plain_text_export(pages_data)
        st.download_button(
            label="📝 Plain Text",
            data=text_bytes,
            file_name=f"{stem}_translated.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Primary format (PDF / DOCX / extra text) ───────────────────────────
    with col3:
        if output_format == "Translated PDF":
            with st.spinner("Building translated PDF …"):
                try:
                    from reconstruction.pdf_builder import build_pdf
                    pdf_bytes = build_pdf(pages_data, page_images)
                    st.download_button(
                        label="📄 Translated PDF",
                        data=pdf_bytes,
                        file_name=f"{stem}_translated.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")
                    logger.exception(exc)

        elif output_format == "Translated DOCX":
            with st.spinner("Building DOCX …"):
                try:
                    from reconstruction.docx_builder import build_docx
                    docx_bytes = build_docx(pages_data, page_images)
                    st.download_button(
                        label="📝 Translated DOCX",
                        data=docx_bytes,
                        file_name=f"{stem}_translated.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"
                        ),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"DOCX generation failed: {exc}")
                    logger.exception(exc)

        else:
            # Raw text — already available above; show a second button here too
            st.download_button(
                label="📝 Raw Text (copy)",
                data=build_plain_text_export(pages_data),
                file_name=f"{stem}_raw.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Welcome / help panel
# ---------------------------------------------------------------------------

def render_welcome() -> None:
    st.markdown(
        """
        ## Welcome to the OCR Document Translator

        Upload a **PDF** or **image** (JPG/PNG) to extract text with Tesseract OCR,
        optionally translate Russian content to English offline, and download the result.

        ---
        ### Quick-start checklist

        | Step | What to do |
        |------|-----------|
        | 1 | Install **Tesseract** and the Russian language pack |
        | 2 | Install **Poppler** (required by pdf2image for PDF support) |
        | 3 | Install **Argos RU→EN model** (see sidebar warning if missing) |
        | 4 | Upload your document using the widget above |
        | 5 | Adjust settings in the sidebar |
        | 6 | Click **Run OCR & Translate** |

        ---
        ### Supported file types
        - **PDF** — multi-page, scanned, or native (with embedded images)
        - **JPG / JPEG / PNG** — single-page image documents

        ### Supported OCR languages
        English · Russian · German · French · Italian · Czech

        ### Translation
        Russian text is detected automatically and translated **completely offline**
        using [Argos Translate](https://github.com/argosopentech/argos-translate).
        No internet connection is required once the model is installed.
        """
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📄 OCR Document Translator")
    st.markdown(
        "Offline OCR · Russian → English translation · "
        "PDF / DOCX output with layout preservation"
    )

    # Dependency check (cached)
    with st.spinner("Checking dependencies …"):
        issues = _check_all_dependencies()

    critical = render_dependency_warnings(issues)
    if critical:
        st.stop()

    # Sidebar
    settings = render_sidebar()

    # File upload
    uploaded = st.file_uploader(
        "Upload a PDF or image file",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Maximum recommended size: ~50 MB. Larger files may be slow.",
        label_visibility="visible",
    )

    if uploaded is None:
        render_welcome()
        return

    file_bytes = uploaded.read()
    filename = uploaded.name
    file_ext = Path(filename).suffix.lower()

    st.markdown(
        f"**File:** `{filename}` &nbsp;|&nbsp; "
        f"**Size:** {len(file_bytes) / 1024:.1f} KB"
    )

    # Load document
    with st.spinner("Loading document …"):
        try:
            if file_ext == ".pdf":
                from utils.file_handler import pdf_to_images, extract_embedded_images
                page_images = pdf_to_images(file_bytes, dpi=settings["dpi"])
                embedded_images = extract_embedded_images(file_bytes)
                st.success(
                    f"✅ PDF loaded — {len(page_images)} page(s), "
                    f"{len(embedded_images)} embedded image(s)"
                )
            else:
                from utils.file_handler import load_image
                page_images = [load_image(file_bytes, filename)]
                embedded_images = []
                w, h = page_images[0].width, page_images[0].height
                st.success(f"✅ Image loaded — {w} × {h} px")
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")
            logger.exception(exc)
            return

    # Run button
    run_clicked = st.button("🔍 Run OCR & Translate", type="primary", use_container_width=True)

    if run_clicked:
        all_pages_data: List[Dict[str, Any]] = []
        progress_bar = st.progress(0.0)
        status = st.empty()

        for page_idx, page_img in enumerate(page_images):
            status.text(
                f"Processing page {page_idx + 1} / {len(page_images)} …"
            )
            try:
                page_data = process_single_page(
                    page_img=page_img,
                    embedded_images=embedded_images,
                    page_num=page_idx,
                    lang=settings["ocr_lang"],
                    confidence_threshold=settings["confidence_threshold"],
                    preprocess_opts=settings["preprocess_opts"],
                    translate_enabled=settings["translate_enabled"],
                )
                all_pages_data.append(page_data)
            except Exception as exc:
                st.error(f"Error on page {page_idx + 1}: {exc}")
                logger.exception(exc)
            finally:
                progress_bar.progress((page_idx + 1) / len(page_images))

        status.text("✅ All pages processed.")
        progress_bar.empty()

        # Cache results in session state
        st.session_state["pages_data"] = all_pages_data
        st.session_state["page_images"] = page_images
        st.session_state["filename"] = filename
        st.session_state["settings"] = settings

    # Render results (from session state so they persist across re-renders)
    if st.session_state.get("pages_data"):
        pages_data: List[Dict[str, Any]] = st.session_state["pages_data"]
        page_images_cached: List[Image.Image] = st.session_state["page_images"]
        cached_filename: str = st.session_state["filename"]
        cached_settings: Dict[str, Any] = st.session_state.get("settings", settings)

        st.markdown("---")
        st.subheader("📊 Results Summary")
        render_summary_metrics(pages_data)

        st.markdown("---")
        st.subheader("📄 Page-by-Page Results")
        for page_data, page_img in zip(pages_data, page_images_cached):
            render_page_results(page_data, page_img, cached_settings)

        render_download_section(
            filename=cached_filename,
            pages_data=pages_data,
            page_images=page_images_cached,
            output_format=settings["output_format"],
        )


if __name__ == "__main__":
    main()
