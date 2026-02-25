"""
OCR Preprocessing Module
Provides image enhancement functions to improve OCR accuracy.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or RGBA image to grayscale."""
    if len(image.shape) == 2:
        return image  # already grayscale
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_threshold(image: np.ndarray) -> np.ndarray:
    """
    Binarize image using Otsu's method on the grayscale channel.
    Returns a single-channel binary image.
    """
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Apply adaptive Gaussian thresholding — better for uneven lighting."""
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means denoising.
    Works on both colour and grayscale images.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = to_grayscale(image)
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew angle in a document image.
    Uses the minimum-area bounding rectangle of dark pixels.
    """
    gray = to_grayscale(image)

    # Find coordinates of non-white (dark) pixels
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return image  # not enough data to estimate angle

    angle = cv2.minAreaRect(coords)[-1]  # angle in [-90, 0)

    if angle < -45:
        angle = 90 + angle
    else:
        angle = -angle  # convert to clockwise rotation

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.debug(f"Deskewed image by {angle:.2f}°")
    return rotated


def sharpen(image: np.ndarray) -> np.ndarray:
    """Apply an unsharp-mask sharpening kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def preprocess_image(
    image: np.ndarray,
    grayscale: bool = False,
    threshold: bool = False,
    denoise_img: bool = False,
    deskew_img: bool = False,
) -> np.ndarray:
    """
    Apply a configurable pipeline of preprocessing steps.

    Parameters
    ----------
    image:       Input image as a NumPy array (BGR or grayscale).
    grayscale:   Convert to grayscale before OCR.
    threshold:   Apply Otsu binarisation.
    denoise_img: Apply Non-Local Means denoising.
    deskew_img:  Correct document skew.

    Returns
    -------
    Preprocessed NumPy array suitable for pytesseract.
    """
    result = image.copy()

    if deskew_img:
        result = deskew(result)

    if denoise_img:
        result = denoise(result)

    if threshold:
        result = apply_threshold(result)
    elif grayscale:
        result = to_grayscale(result)

    return result


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV-compatible NumPy array (RGB → BGR)."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(array: np.ndarray) -> Image.Image:
    """Convert OpenCV NumPy array to PIL Image."""
    if len(array.shape) == 2:
        return Image.fromarray(array, mode="L")
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
