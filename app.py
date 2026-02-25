"""
OCR Document Translator — Main Streamlit Application
=====================================================
Offline OCR (pytesseract) with Russian → English translation (Argos Translate).
Supports PDF and image inputs; outputs translated PDF, DOCX, or plain text.

Startup behaviour
-----------------
1. Missing Python packages are pip-installed automatically on first run.
2. Tesseract is located at common install paths (no PATH edit required on Windows).
3. Poppler (pdf2image backend) is located automatically on Windows.
4. The Argos RU→EN model can be downloaded with one click inside the app.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    div[data-testid="stExpander"] > div { padding: 0.5rem 1rem; }
    .translated-badge {
        background: #d4edda; color: #155724;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.78rem; font-weight: 600;
    }
    .ok-badge   { color: #28a745; font-weight: 700; }
    .warn-badge { color: #ffc107; font-weight: 700; }
    .err-badge  { color: #dc3545; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# ① Startup: auto-install Python packages + configure system binaries
#    NOT cached — re-runs whenever the user clicks Retry or supplies a path.
# ---------------------------------------------------------------------------

def _run_startup(
    custom_tesseract: Optional[str] = None,
    custom_poppler: Optional[str] = None,
) -> Any:
    from utils.installer import run_startup_checks
    return run_startup_checks(
        custom_tesseract_path=custom_tesseract or None,
        custom_poppler_path=custom_poppler or None,
    )


def _show_startup_banner(result: Any) -> bool:
    """
    Render the dependency status panel.
    Returns True when Tesseract is still missing (caller should st.stop()).
    """
    from utils.installer import poppler_install_hint

    ok_count = sum([result.tesseract_ok, result.argos_ok, result.poppler_ok])
    label = f"🔧 Dependency Status  ({ok_count}/3 ready)"

    with st.expander(label, expanded=not result.all_ok):

        # ── Python packages ────────────────────────────────────────────────
        if result.python_packages_installed:
            good = [p for p, ok, _ in result.python_packages_installed if ok]
            bad  = [(p, m) for p, ok, m in result.python_packages_installed if not ok]
            if good:
                st.success(f"✅ Auto-installed: {', '.join(good)}")
            for pkg, msg in bad:
                st.error(f"❌ Could not install `{pkg}`: {msg}")
        else:
            st.success("✅ All Python packages already installed.")

        st.markdown("---")

        # ── Tesseract ──────────────────────────────────────────────────────
        if result.tesseract_ok:
            st.success(f"✅ Tesseract — {result.tesseract_message}")
        else:
            st.error(
                "❌ **Tesseract OCR not found.**  "
                "Install it, then click **🔄 Retry Detection** below."
            )
            st.code(result.tesseract_message, language="bash")

            # Manual path input
            st.markdown("**Already installed? Enter the path manually:**")
            col_path, col_btn = st.columns([3, 1])
            with col_path:
                manual_tess = st.text_input(
                    "Tesseract executable path",
                    placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    key="manual_tess_path",
                    label_visibility="collapsed",
                )
            with col_btn:
                if st.button("Apply", key="apply_tess_path"):
                    st.session_state["custom_tesseract"] = manual_tess
                    st.session_state["startup_done"] = False
                    st.rerun()

        st.markdown("---")

        # ── Poppler ────────────────────────────────────────────────────────
        if result.poppler_ok:
            loc = f" at `{result.poppler_path}`" if result.poppler_path else " (in PATH)"
            st.success(f"✅ Poppler{loc}")
        else:
            st.warning("⚠️ **Poppler not found** — PDF upload will fail. Image files still work.")
            st.code(poppler_install_hint(), language="bash")

            col_ppath, col_pbtn = st.columns([3, 1])
            with col_ppath:
                manual_pop = st.text_input(
                    "Poppler bin\\ path",
                    placeholder=r"C:\poppler\bin",
                    key="manual_pop_path",
                    label_visibility="collapsed",
                )
            with col_pbtn:
                if st.button("Apply", key="apply_pop_path"):
                    st.session_state["custom_poppler"] = manual_pop
                    st.session_state["startup_done"] = False
                    st.rerun()

        st.markdown("---")

        # ── Argos model ────────────────────────────────────────────────────
        if result.argos_ok:
            st.success("✅ Argos RU→EN model ready.")
        else:
            st.warning(
                "⚠️ **RU→EN translation model not installed.**  "
                "OCR still works — Russian text just won't be translated."
            )
            _render_argos_install_button()

        st.markdown("---")

        # ── Retry button ───────────────────────────────────────────────────
        if st.button("🔄 Retry Detection", help="Re-scan for Tesseract and Poppler after installing them."):
            st.session_state["startup_done"] = False
            st.rerun()

    return result.tesseract_missing


def _render_argos_install_button() -> None:
    """One-click download and install of the Argos RU→EN model."""
    st.markdown(
        "The model is **~100 MB** and needs internet **once only**. "
        "After that the app is fully offline."
    )
    if st.button("⬇️ Download & Install RU→EN Model Now", type="primary"):
        status_box = st.empty()
        bar = st.progress(0)
        calls: List[str] = []

        def _cb(msg: str) -> None:
            calls.append(msg)
            status_box.info(f"⏳ {msg}")
            bar.progress(min(1.0, len(calls) / 3))

        from utils.installer import install_argos_model
        ok, msg = install_argos_model(progress_callback=_cb)
        bar.progress(1.0)
        if ok:
            status_box.success(f"✅ {msg}")
            st.session_state["startup_done"] = False
            st.rerun()
        else:
            status_box.error(f"❌ {msg}")
            st.code(
                "# Run this once in any terminal / PowerShell:\n"
                'python -c "\n'
                "import argostranslate.package\n"
                "argostranslate.package.update_package_index()\n"
                "pkgs = argostranslate.package.get_available_packages()\n"
                "pkg  = next(p for p in pkgs if p.from_code=='ru' and p.to_code=='en')\n"
                'argostranslate.package.install_from_path(pkg.download())\n"',
                language="bash",
            )


# ---------------------------------------------------------------------------
# ② Processing pipeline
# ---------------------------------------------------------------------------

def _draw_bounding_boxes(
    image: Image.Image,
    blocks: List[Dict[str, Any]],
) -> Image.Image:
    """Overlay coloured bounding boxes. Green = kept, Red = translated."""
    img_copy = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for block in blocks:
        x, y, w, h = block["bbox"]
        was_translated = block.get("was_translated", False)
        fill    = (220, 53, 69, 110) if was_translated else (40, 167, 69,  90)
        outline = (220, 53, 69, 255) if was_translated else (40, 167, 69, 255)
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline, width=2)

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
    """Full pipeline: preprocess → OCR → layout analysis → translate."""
    from ocr.engine import ocr_to_blocks, run_ocr
    from ocr.layout import detect_columns, sort_blocks_reading_order
    from ocr.preprocessing import cv2_to_pil, pil_to_cv2, preprocess_image
    from translation.translator import translate_blocks
    from utils.file_handler import get_page_dimensions

    t0 = time.perf_counter()

    # Preprocessing
    cv2_img = pil_to_cv2(page_img)
    processed_cv2 = preprocess_image(
        cv2_img,
        grayscale=preprocess_opts.get("grayscale", False),
        threshold=preprocess_opts.get("threshold", False),
        denoise_img=preprocess_opts.get("denoise", False),
        deskew_img=preprocess_opts.get("deskew", False),
    )
    processed_pil = cv2_to_pil(processed_cv2)

    # OCR
    ocr_df = run_ocr(processed_pil, lang=lang, confidence_threshold=confidence_threshold)
    blocks = ocr_to_blocks(ocr_df)

    # Layout
    page_w, page_h = get_page_dimensions(page_img)
    num_cols = detect_columns(blocks, page_w)
    blocks = sort_blocks_reading_order(blocks, num_cols, page_w)

    # Translation
    blocks = translate_blocks(blocks, enabled=translate_enabled)

    page_images_embedded = [img for img in embedded_images if img.get("page") == page_num]

    return {
        "page_number": page_num + 1,
        "width": page_w,
        "height": page_h,
        "blocks": blocks,
        "images": page_images_embedded,
        "num_columns": num_cols,
        "processing_time": time.perf_counter() - t0,
        "word_count": sum(len(b["text"].split()) for b in blocks),
        "translation_word_count": sum(
            len(b.get("translated_text", "").split())
            for b in blocks
            if b.get("was_translated")
        ),
    }


# ---------------------------------------------------------------------------
# ③ Export helpers
# ---------------------------------------------------------------------------

def _build_json_export(filename: str, pages_data: List[Dict[str, Any]]) -> bytes:
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


def _build_plain_text_export(pages_data: List[Dict[str, Any]]) -> bytes:
    lines: List[str] = []
    for page in pages_data:
        lines += [f"{'=' * 60}", f"  Page {page['page_number']}", f"{'=' * 60}"]
        for block in page["blocks"]:
            text = block.get("translated_text", block.get("text", ""))
            if text.strip():
                lines += [text, ""]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# ④ Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")

        # OCR
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
        ocr_lang = lang_map[
            st.selectbox("OCR Language Pack", list(lang_map.keys()), index=0)
        ]
        confidence_threshold = float(
            st.slider("Confidence Threshold (%)", 0, 100, 30, 5,
                      help="Discard OCR words below this score.")
        )
        dpi = int(
            st.select_slider("Rendering DPI (PDF only)", [100, 150, 200, 250, 300], value=200,
                             help="Higher = better quality, more RAM.")
        )

        st.markdown("---")

        # Preprocessing
        st.subheader("🖼️ Preprocessing")
        preprocess_opts = {
            "grayscale": st.checkbox("Grayscale"),
            "threshold": st.checkbox("Binarize (Otsu)", help="Best for scanned black/white docs."),
            "denoise":   st.checkbox("Denoise"),
            "deskew":    st.checkbox("Deskew",   help="Auto-correct skewed pages."),
        }

        st.markdown("---")

        # Translation
        st.subheader("🌐 Translation")
        translate_enabled = st.checkbox("Enable RU → EN Translation", value=True)

        st.markdown("---")

        # Display
        st.subheader("🖥️ Display")
        show_bbox       = st.checkbox("Show Bounding Boxes",
                                      help="Green = kept  |  Red = translated")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)

        st.markdown("---")

        # Output
        st.subheader("📤 Output Format")
        output_format = st.radio(
            "Generate",
            ["Translated PDF", "Translated DOCX", "Raw Text Only"],
            index=0,
        )

    return dict(
        ocr_lang=ocr_lang,
        confidence_threshold=confidence_threshold,
        dpi=dpi,
        preprocess_opts=preprocess_opts,
        translate_enabled=translate_enabled,
        show_bbox=show_bbox,
        show_confidence=show_confidence,
        output_format=output_format,
    )


# ---------------------------------------------------------------------------
# ⑤ Results display
# ---------------------------------------------------------------------------

def _render_metrics(pages_data: List[Dict[str, Any]]) -> None:
    total_words      = sum(p["word_count"] for p in pages_data)
    total_translated = sum(p["translation_word_count"] for p in pages_data)
    total_time       = sum(p["processing_time"] for p in pages_data)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pages",            len(pages_data))
    c2.metric("Words Detected",   f"{total_words:,}")
    c3.metric("Words Translated",  f"{total_translated:,}")
    c4.metric("Processing Time",  f"{total_time:.1f} s")
    c5.metric(
        "Translation %",
        f"{total_translated / total_words * 100:.0f}%" if total_words else "—",
    )


def _render_page(
    page_data: Dict[str, Any],
    page_img: Image.Image,
    settings: Dict[str, Any],
) -> None:
    blocks   = page_data["blocks"]
    page_num = page_data["page_number"]

    with st.expander(
        f"📄 Page {page_num}  —  "
        f"{page_data['word_count']} words  |  "
        f"{page_data['num_columns']} col(s)  |  "
        f"{page_data['processing_time']:.2f} s",
        expanded=(page_num == 1),
    ):
        col_img, col_txt = st.columns([1, 1], gap="medium")

        with col_img:
            st.markdown("**Original Page**")
            display = (
                _draw_bounding_boxes(page_img, blocks)
                if settings["show_bbox"] and blocks
                else page_img
            )
            st.image(display, use_column_width=True)
            if settings["show_bbox"] and blocks:
                st.caption("🟢 Kept original  🔴 Translated from Russian")
            if page_data["num_columns"] > 1:
                st.info(f"📊 Multi-column layout ({page_data['num_columns']} columns)")
            st.caption(
                f"{page_data['width']} × {page_data['height']} px  |  "
                f"{len(page_data.get('images', []))} embedded image(s)"
            )

        with col_txt:
            st.markdown("**Extracted & Translated Text**")
            if not blocks:
                st.info("No text detected on this page.")
            else:
                for block in blocks:
                    original       = block.get("text", "")
                    translated     = block.get("translated_text", original)
                    was_translated = block.get("was_translated", False)
                    conf           = block.get("confidence", 0.0)
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


def _render_downloads(
    filename: str,
    pages_data: List[Dict[str, Any]],
    page_images: List[Image.Image],
    output_format: str,
) -> None:
    st.markdown("---")
    st.subheader("⬇️ Download Results")
    stem = Path(filename).stem
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "📊 JSON (OCR data)",
            data=_build_json_export(filename, pages_data),
            file_name=f"{stem}_ocr.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📝 Plain Text",
            data=_build_plain_text_export(pages_data),
            file_name=f"{stem}_translated.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with c3:
        if output_format == "Translated PDF":
            with st.spinner("Building PDF …"):
                try:
                    from reconstruction.pdf_builder import build_pdf
                    st.download_button(
                        "📄 Translated PDF",
                        data=build_pdf(pages_data, page_images),
                        file_name=f"{stem}_translated.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")

        elif output_format == "Translated DOCX":
            with st.spinner("Building DOCX …"):
                try:
                    from reconstruction.docx_builder import build_docx
                    st.download_button(
                        "📝 Translated DOCX",
                        data=build_docx(pages_data, page_images),
                        file_name=f"{stem}_translated.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document"
                        ),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"DOCX generation failed: {exc}")
        else:
            st.download_button(
                "📝 Raw Text",
                data=_build_plain_text_export(pages_data),
                file_name=f"{stem}_raw.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# ⑥ Welcome panel
# ---------------------------------------------------------------------------

def _render_welcome() -> None:
    st.markdown(
        """
        ## Welcome to the OCR Document Translator

        Upload a **PDF** or **image** (JPG / PNG) to:

        - Extract text using **Tesseract OCR** (word-level bounding boxes)
        - Automatically detect and translate **Russian → English** (offline)
        - Preserve the original document layout and embedded images
        - Download the result as a translated **PDF**, **DOCX**, or plain text

        ---

        ### Supported inputs
        | Format | Notes |
        |--------|-------|
        | PDF    | Scanned or native; multi-page; may contain embedded images |
        | JPG / JPEG / PNG | Single-page image |

        ### OCR Languages
        English · Russian · German · French · Italian · Czech

        ### How translation works
        Russian text is detected by Cyrillic character ratio (≥ 35 %).
        Translation uses **Argos Translate** — completely offline once the
        model is installed (see the status panel above if the model is missing).
        """
    )


# ---------------------------------------------------------------------------
# ⑦ Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📄 OCR Document Translator")
    st.markdown(
        "Offline OCR · Russian → English · "
        "PDF / DOCX output with layout preservation"
    )

    # ── Startup: auto-install packages + configure binaries ─────────────────
    # Re-runs whenever the user clicks Retry or applies a manual path.
    if not st.session_state.get("startup_done", False):
        with st.spinner("Checking and configuring dependencies …"):
            startup = _run_startup(
                custom_tesseract=st.session_state.get("custom_tesseract", ""),
                custom_poppler=st.session_state.get("custom_poppler", ""),
            )
        st.session_state["startup_result"] = startup
        st.session_state["startup_done"] = True
    else:
        startup = st.session_state["startup_result"]

    # Show status banner; returns True when Tesseract is still missing
    tesseract_missing = _show_startup_banner(startup)

    if tesseract_missing:
        st.stop()

    # Sidebar
    settings = _render_sidebar()

    # File upload
    uploaded = st.file_uploader(
        "Upload a PDF or image file",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Recommended max size: ~50 MB",
    )

    if uploaded is None:
        _render_welcome()
        return

    file_bytes = uploaded.read()
    filename   = uploaded.name
    file_ext   = Path(filename).suffix.lower()

    st.markdown(
        f"**File:** `{filename}`  |  **Size:** {len(file_bytes) / 1024:.1f} KB"
    )

    # Load document
    with st.spinner("Loading document …"):
        try:
            if file_ext == ".pdf":
                from utils.file_handler import extract_embedded_images, pdf_to_images
                page_images     = pdf_to_images(file_bytes, dpi=settings["dpi"])
                embedded_images = extract_embedded_images(file_bytes)
                st.success(
                    f"✅ PDF loaded — {len(page_images)} page(s), "
                    f"{len(embedded_images)} embedded image(s)"
                )
            else:
                from utils.file_handler import load_image
                page_images     = [load_image(file_bytes, filename)]
                embedded_images = []
                w, h = page_images[0].width, page_images[0].height
                st.success(f"✅ Image loaded — {w} × {h} px")
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")
            logger.exception(exc)
            return

    # Run button
    if st.button("🔍 Run OCR & Translate", type="primary", use_container_width=True):
        all_pages: List[Dict[str, Any]] = []
        bar    = st.progress(0.0)
        status = st.empty()
        total  = len(page_images)

        for idx, page_img in enumerate(page_images):
            status.text(f"Processing page {idx + 1} / {total} …")
            try:
                page_data = process_single_page(
                    page_img=page_img,
                    embedded_images=embedded_images,
                    page_num=idx,
                    lang=settings["ocr_lang"],
                    confidence_threshold=settings["confidence_threshold"],
                    preprocess_opts=settings["preprocess_opts"],
                    translate_enabled=settings["translate_enabled"],
                )
                all_pages.append(page_data)
            except Exception as exc:
                st.error(f"Error on page {idx + 1}: {exc}")
                logger.exception(exc)
            bar.progress((idx + 1) / total)

        bar.empty()
        status.success(f"✅ Processed {len(all_pages)} / {total} page(s).")

        st.session_state["pages_data"]   = all_pages
        st.session_state["page_images"]  = page_images
        st.session_state["filename"]     = filename
        st.session_state["settings"]     = settings

    # Render results (persists between sidebar changes without re-running OCR)
    if st.session_state.get("pages_data"):
        pages_data:  List[Dict[str, Any]] = st.session_state["pages_data"]
        imgs_cached: List[Image.Image]    = st.session_state["page_images"]
        fn_cached:   str                  = st.session_state["filename"]
        cfg:         Dict[str, Any]       = st.session_state.get("settings", settings)

        st.markdown("---")
        st.subheader("📊 Summary")
        _render_metrics(pages_data)

        st.markdown("---")
        st.subheader("📄 Pages")
        for pd_, pi_ in zip(pages_data, imgs_cached):
            _render_page(pd_, pi_, cfg)

        _render_downloads(fn_cached, pages_data, imgs_cached, settings["output_format"])


if __name__ == "__main__":
    main()
