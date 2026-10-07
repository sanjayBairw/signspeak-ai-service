"""
SignSpeak AI Service — Preprocessing helpers
============================================

Standalone helpers for decoding and validating frames before they are
passed to the LandmarkExtractor.  Kept in a separate module so they
can be unit-tested independently.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)


def decode_base64_frame(b64_data: str) -> np.ndarray:
    """
    Decode a base64-encoded image string (JPEG / PNG) into a BGR ndarray.

    Strips any data-URI prefix (e.g. "data:image/jpeg;base64,") before decoding.

    Raises:
        ValueError: if the data cannot be decoded into a valid image.
    """
    # Strip data-URI prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(b64_data)
    except Exception as exc:
        raise ValueError(f"[INVALID_FRAME] base64 decode failed: {exc}") from exc

    return decode_image_bytes(raw_bytes)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG / PNG) into a BGR ndarray.

    Raises:
        ValueError: if OpenCV cannot decode the bytes.
    """
    settings = get_settings()
    max_size = settings.max_frame_size_bytes

    if len(image_bytes) == 0:
        raise ValueError("[INVALID_FRAME] Empty image bytes.")

    if len(image_bytes) > max_size:
        raise ValueError(
            f"[INVALID_FRAME] Frame size {len(image_bytes)} bytes exceeds "
            f"limit of {max_size} bytes."
        )

    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("[INVALID_IMAGE] OpenCV could not decode image bytes.")

    return frame


def validate_feature_vector(features: np.ndarray) -> bool:
    """Return True if features is a valid (126,) float32 vector with no NaN/Inf."""
    if features.shape != (126,):
        return False
    if not np.isfinite(features).all():
        return False
    return True
