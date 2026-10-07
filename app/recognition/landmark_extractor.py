"""
SignSpeak AI Service — MediaPipe Landmark Extractor
====================================================

Wraps MediaPipe HandLandmarker (IMAGE mode) and produces exactly the
same 126-feature vector that extract_landmarks.py produced during training:

    2 hands × 21 landmarks × (x, y, z)
    ↓
    wrist-relative coordinates
    ↓
    hands sorted left-to-right by wrist X
    ↓
    missing hand stays as zeros
    ↓
    flatten → 126 values

CRITICAL: IMAGE mode + detect() is used, matching training exactly.
VIDEO/LIVE_STREAM mode applies temporal smoothing that training never saw.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config import get_settings

logger = logging.getLogger(__name__)

# Number of landmarks per hand (MediaPipe hand model)
_LANDMARKS_PER_HAND = 21
# XYZ = 3 coordinates
_COORDS = 3


class LandmarkExtractor:
    """
    Holds a single shared MediaPipe HandLandmarker instance (IMAGE mode).

    Usage:
        extractor = LandmarkExtractor()
        extractor.load()
        features = extractor.extract_from_frame(bgr_frame)  # → (126,) or None
    """

    def __init__(self) -> None:
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Create the HandLandmarker. Call once at application startup."""
        settings = get_settings()

        if not os.path.exists(settings.hand_model_path):
            raise FileNotFoundError(
                f"[MODEL_NOT_FOUND] MediaPipe hand model not found. "
                f"Set HAND_MODEL_PATH environment variable. "
                f"Path checked: {settings.hand_model_path}"
            )

        logger.info("Loading MediaPipe HandLandmarker…")

        # Try GPU delegate first, fall back to CPU
        # (mirrors realtime_test.py behaviour for best performance)
        loaded_with = "CPU"
        try:
            gpu_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=settings.hand_model_path,
                    delegate=python.BaseOptions.Delegate.GPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=settings.num_hands,
                min_hand_detection_confidence=settings.min_hand_detection_confidence,
                min_hand_presence_confidence=settings.min_hand_presence_confidence,
                min_tracking_confidence=settings.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(gpu_options)
            loaded_with = "GPU"
            logger.info("MediaPipe HandLandmarker ready (GPU delegate).")
        except Exception as gpu_err:
            logger.warning(
                "GPU delegate unavailable (%s). Falling back to CPU.",
                repr(gpu_err),
            )
            cpu_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=settings.hand_model_path,
                    delegate=python.BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=settings.num_hands,
                min_hand_detection_confidence=settings.min_hand_detection_confidence,
                min_hand_presence_confidence=settings.min_hand_presence_confidence,
                min_tracking_confidence=settings.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(cpu_options)
            logger.info("MediaPipe HandLandmarker ready (CPU delegate).")

        self._loaded = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_preserving_aspect(self, frame: np.ndarray, target_width: int) -> np.ndarray:
        """
        Scale frame so that width == target_width, preserving aspect ratio.
        Matches extract_landmarks.py — only applied when frame is wider than target.
        """
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame
        new_h = int(h * target_width / w)
        return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

    def _build_feature(self, result) -> tuple[np.ndarray, int]:
        """
        Convert MediaPipe result → (126,) feature vector and hand count.

        Feature layout (matches training):
            slot 0 = hand with smaller wrist.x  (left-most in image)
            slot 1 = hand with larger  wrist.x  (right-most in image)
            each slot = 21 landmarks × (x - wrist.x, y - wrist.y, z - wrist.z)
        """
        settings = get_settings()
        num_hands = settings.num_hands

        # Shape: (num_hands, 21, 3) — zero = no hand
        feature_arr = np.zeros((num_hands, _LANDMARKS_PER_HAND, _COORDS), dtype=np.float32)

        hands = result.hand_landmarks
        hands_detected = 0

        if hands:
            # Sort by wrist X to get stable left-to-right order
            sorted_hands = sorted(hands, key=lambda h: h[0].x)
            hands_detected = len(sorted_hands)

            for hand_id, hand in enumerate(sorted_hands[:num_hands]):
                wrist = hand[0]
                for j, lm in enumerate(hand):
                    feature_arr[hand_id, j, 0] = lm.x - wrist.x
                    feature_arr[hand_id, j, 1] = lm.y - wrist.y
                    feature_arr[hand_id, j, 2] = lm.z - wrist.z

        # Flatten → (126,)
        return feature_arr.reshape(-1), hands_detected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def extract_from_bgr(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Run MediaPipe on a BGR frame and return (features, hands_detected).

        features: float32 ndarray of shape (126,)
        hands_detected: int, 0–2

        Raises:
            RuntimeError: if the extractor is not loaded.
            ValueError: if the frame is invalid.
        """
        if not self._loaded or self._landmarker is None:
            raise RuntimeError("[MEDIAPIPE_ERROR] LandmarkExtractor not loaded.")

        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("[INVALID_IMAGE] Empty or None frame received.")

        settings = get_settings()

        # 1. Resize preserving aspect ratio (matches training)
        frame = self._resize_preserving_aspect(bgr_frame, settings.inference_width)

        # 2. BGR → RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 4. Detect — IMAGE mode, single call (matches training exactly)
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as exc:
            raise RuntimeError(f"[MEDIAPIPE_ERROR] Detection failed: {exc}") from exc

        # 5. Build 126-feature vector
        features, hands_detected = self._build_feature(result)
        return features, hands_detected

    def extract_from_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, int]:
        """
        Decode image bytes (JPEG / PNG) then call extract_from_bgr.

        Raises:
            ValueError: if decoding fails.
        """
        import numpy as np
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("[INVALID_IMAGE] Could not decode image bytes.")
        return self.extract_from_bgr(frame)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

extractor = LandmarkExtractor()
