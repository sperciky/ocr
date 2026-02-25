"""
Layout Analysis Module
Detects multi-column layouts and produces a correct reading order
for OCR blocks extracted from a document page.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _block_center_x(block: Dict[str, Any]) -> float:
    x, _, w, _ = block["bbox"]
    return x + w / 2


def _block_center_y(block: Dict[str, Any]) -> float:
    _, y, _, h = block["bbox"]
    return y + h / 2


def detect_columns(blocks: List[Dict[str, Any]], page_width: int) -> int:
    """
    Heuristic column count detection (supports 1 or 2 columns).

    Strategy
    --------
    Project each block's horizontal centre onto the x-axis, then check
    whether blocks cluster significantly in both halves of the page.
    If both halves contain at least 25 % of all blocks, we call it 2-column.

    Returns
    -------
    1 or 2
    """
    if not blocks or page_width == 0:
        return 1

    mid = page_width / 2
    x_centers = [_block_center_x(b) for b in blocks]

    left_count = sum(1 for x in x_centers if x < mid)
    right_count = sum(1 for x in x_centers if x >= mid)
    total = len(x_centers)

    threshold = 0.25  # each side must have at least 25 % of blocks
    if (left_count / total) >= threshold and (right_count / total) >= threshold:
        logger.debug(
            f"Two-column layout detected "
            f"(left={left_count}, right={right_count}, total={total})"
        )
        return 2

    return 1


def sort_blocks_reading_order(
    blocks: List[Dict[str, Any]],
    num_columns: int = 1,
    page_width: int = 0,
) -> List[Dict[str, Any]]:
    """
    Return blocks sorted in natural reading order.

    For a single-column layout: top-to-bottom, left-to-right.
    For a two-column layout:    left column top-to-bottom first,
                                then right column top-to-bottom.

    Parameters
    ----------
    blocks:      List of block dicts (each must have a 'bbox' key).
    num_columns: Result of detect_columns().
    page_width:  Width of the page in pixels (required for 2-column split).
    """
    if not blocks:
        return []

    if num_columns == 1 or page_width == 0:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    mid = page_width / 2
    left_col = [b for b in blocks if _block_center_x(b) < mid]
    right_col = [b for b in blocks if _block_center_x(b) >= mid]

    left_sorted = sorted(left_col, key=lambda b: b["bbox"][1])
    right_sorted = sorted(right_col, key=lambda b: b["bbox"][1])

    return left_sorted + right_sorted


def group_into_paragraphs(
    blocks: List[Dict[str, Any]],
    line_height_ratio: float = 0.8,
) -> List[List[Dict[str, Any]]]:
    """
    Group consecutive vertically-close blocks into paragraphs.

    Two blocks are in the same paragraph when the vertical gap between
    them is less than `line_height_ratio` × average block height.

    Returns
    -------
    List of paragraphs, each a list of block dicts.
    """
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])
    paragraphs: List[List[Dict[str, Any]]] = [[sorted_blocks[0]]]

    for block in sorted_blocks[1:]:
        prev_block = paragraphs[-1][-1]
        prev_bottom = prev_block["bbox"][1] + prev_block["bbox"][3]
        curr_top = block["bbox"][1]
        gap = curr_top - prev_bottom

        avg_height = float(
            np.mean([b["bbox"][3] for b in paragraphs[-1]])
        )
        if gap > avg_height * line_height_ratio:
            paragraphs.append([block])
        else:
            paragraphs[-1].append(block)

    return paragraphs


def estimate_font_size(block: Dict[str, Any], dpi: int = 200) -> float:
    """
    Estimate the approximate font size in points from the block bounding box.

    Tesseract returns pixel coordinates at the image DPI.
    1 point = 1/72 inch, so:
        font_pt = block_height_px / lines / (dpi / 72)
    """
    _, _, _, h = block["bbox"]
    words = block.get("words", [])
    if not words:
        return 10.0

    # Estimate line count by unique top values (rounded to nearest 5 px)
    tops = {round(w["top"] / 5) * 5 for w in words}
    num_lines = max(1, len(tops))

    line_height_px = h / num_lines
    font_pt = line_height_px / (dpi / 72)
    return max(6.0, min(72.0, font_pt))
