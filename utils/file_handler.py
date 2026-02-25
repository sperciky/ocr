"""
File Handler Utilities
Handles loading PDFs and images, DPI-aware coordinate scaling,
and extraction of embedded images from PDF files.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_PDF_EXTENSION = ".pdf"


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 200,
) -> List[Image.Image]:
    """
    Rasterise every page of a PDF to a PIL Image.

    Parameters
    ----------
    pdf_bytes : Raw PDF file bytes.
    dpi       : Rendering resolution. Higher = better OCR, more memory.

    Returns
    -------
    Ordered list of PIL Images (one per page), all in RGB mode.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        raise ImportError(
            "pdf2image is not installed. Run: pip install pdf2image\n"
            "Also install Poppler:\n"
            "  macOS  : brew install poppler\n"
            "  Linux  : sudo apt-get install poppler-utils\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows"
        ) from exc

    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="RGB")
        logger.info(f"Converted PDF to {len(images)} image(s) at {dpi} DPI")
        return images
    except Exception as exc:
        raise RuntimeError(f"Failed to rasterise PDF: {exc}") from exc


def extract_embedded_images(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract raster images embedded inside a PDF using PyMuPDF.

    Each returned dict contains:
        page  : int          — 0-based page index
        index : int          — image index on that page
        image : PIL.Image    — the extracted image
        bbox  : [x, y, w, h] — position in PDF user-space points

    Falls back to an empty list when PyMuPDF is not installed.
    """
    results: List[Dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not found — embedded image extraction skipped. "
            "Install with: pip install pymupdf"
        )
        return results

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx, page in enumerate(doc):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as exc:
                    logger.debug(f"Skipping image xref={xref}: {exc}")
                    continue

                # Position on the page (PDF points, origin = bottom-left)
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    # Convert to top-left origin: y = page_height - rect_y1
                    bbox = [
                        float(r.x0),
                        float(page.rect.height - r.y1),
                        float(r.width),
                        float(r.height),
                    ]
                else:
                    bbox = [0.0, 0.0, float(pil_img.width), float(pil_img.height)]

                results.append(
                    {
                        "page": page_idx,
                        "index": img_idx,
                        "image": pil_img,
                        "bbox": bbox,
                    }
                )

        doc.close()
        logger.info(
            f"Extracted {len(results)} embedded image(s) from PDF"
        )
    except Exception as exc:
        logger.error(f"Embedded image extraction error: {exc}")

    return results


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_bytes: bytes, filename: str = "") -> Image.Image:
    """
    Load an image from raw bytes and return it as an RGB PIL Image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load image '{filename}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Coordinate scaling helpers
# ---------------------------------------------------------------------------

def scale_bbox(
    bbox: List[float],
    scale_x: float,
    scale_y: float,
) -> List[int]:
    """Scale a [x, y, w, h] bounding box by independent x/y factors."""
    x, y, w, h = bbox
    return [
        int(x * scale_x),
        int(y * scale_y),
        int(w * scale_x),
        int(h * scale_y),
    ]


def get_dpi_scale(dpi: int = 200) -> float:
    """
    Return the scale factor from PDF points (72 pt/inch) to pixels at *dpi*.
    e.g. at 200 DPI: scale = 200/72 ≈ 2.78
    """
    return dpi / 72.0


# ---------------------------------------------------------------------------
# NumPy / PIL conversion helpers (used across modules)
# ---------------------------------------------------------------------------

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to a NumPy uint8 array in RGB order."""
    return np.array(image.convert("RGB"))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if len(array.shape) == 2:
        return Image.fromarray(array, mode="L")
    return Image.fromarray(array)


def get_page_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Return (width, height) of a PIL Image."""
    return image.width, image.height


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def is_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() == SUPPORTED_PDF_EXTENSION


def is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
