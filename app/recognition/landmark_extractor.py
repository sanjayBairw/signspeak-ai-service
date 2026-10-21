
"""
SignSpeak - Hand Landmark Extractor
Extracts 126 hand landmark features (2 hands x 21 landmarks x 3 coordinates)
using MediaPipe Tasks HandLandmarker.

Render deployment: CPU delegate only.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.config import get_settings

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """Extract normalized hand landmarks from video frames."""

    def __init__(self):
        self.settings = get_settings()
        self._landmarker = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return whether the MediaPipe landmarker is initialized."""
        return self._loaded and self._landmarker is not None

    def load(self) -> None:
        """Initialize MediaPipe HandLandmarker using the CPU delegate."""
        self.settings = get_settings()
        if self.is_loaded:
            logger.info("MediaPipe HandLandmarker already initialized.")
            return

        model_path = Path(self.settings.hand_model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {model_path}"
            )

        logger.info("Initializing MediaPipe HandLandmarker (CPU)...")
        start_time = time.time()

        try:
            base_options = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )

            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.settings.num_hands,
                min_hand_detection_confidence=(
                    self.settings.min_hand_detection_confidence
                ),
                min_hand_presence_confidence=(
                    self.settings.min_hand_presence_confidence
                ),
                min_tracking_confidence=(
                    self.settings.min_tracking_confidence
                ),
            )

            self._landmarker = (
                vision.HandLandmarker.create_from_options(options)
            )
            self._loaded = True

            logger.info(
                "MediaPipe HandLandmarker ready in %.2f seconds.",
                time.time() - start_time,
            )

        except Exception:
            self._landmarker = None
            self._loaded = False
            logger.exception(
                "Failed to initialize MediaPipe HandLandmarker."
            )
            raise

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                logger.exception(
                    "Error while closing MediaPipe HandLandmarker."
                )
            finally:
                self._landmarker = None
                self._loaded = False
                logger.info("MediaPipe HandLandmarker closed.")

    def extract_from_bgr(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Extract 126 features and detected hand count from a BGR image frame.

        Returns:
            Tuple of (features, hands_detected):
                - features: float32 array of shape (126,).
                  Each hand contributes 21 landmarks x (x, y, z).
                  Missing hands are represented by zeros.
                - hands_detected: int count of detected hands (0, 1, or 2).
        """
        if not self.is_loaded:
            raise RuntimeError(
                "LandmarkExtractor is not loaded. Call load() first."
            )

        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid NumPy image.")

        if frame.size == 0:
            raise ValueError("Input frame is empty.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected a three-channel BGR image with shape (H, W, 3)."
            )

        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Invalid frame dimensions.")

        # Resize to reduce inference cost while preserving aspect ratio.
        inference_width = int(self.settings.inference_width)

        if inference_width > 0 and width > inference_width:
            scale = inference_width / float(width)
            new_height = max(1, int(round(height * scale)))

            frame = cv2.resize(
                frame,
                (inference_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        # MediaPipe expects RGB images.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self._landmarker.detect(mp_image)

        # Always return exactly 2 * 21 * 3 = 126 features.
        features = np.zeros((2, 21, 3), dtype=np.float32)

        if not result.hand_landmarks:
            return features.reshape(-1), 0

        detected_hands = []

        for hand_landmarks in result.hand_landmarks:
            if len(hand_landmarks) != 21:
                logger.warning(
                    "Expected 21 landmarks, received %d.",
                    len(hand_landmarks),
                )
                continue

            hand = np.array(
                [
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in hand_landmarks
                ],
                dtype=np.float32,
            )

            # Make coordinates relative to the wrist.
            # This reduces sensitivity to the hand's position in the frame.
            hand -= hand[0].copy()

            # Sort hands from left to right for consistent feature ordering.
            wrist_x = float(hand_landmarks[0].x)
            detected_hands.append((wrist_x, hand))

        detected_hands.sort(key=lambda item: item[0])

        for index, (_, hand) in enumerate(detected_hands[:2]):
            features[index] = hand

        num_hands = min(len(detected_hands), 2)
        return features.reshape(-1), num_hands

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 126 features from a BGR image frame.

        Returns:
            float32 array of shape (126,).
            Each hand contributes 21 landmarks x (x, y, z).
            Missing hands are represented by zeros.
        """
        features, _ = self.extract_from_bgr(frame)
        return features

    def extract_sequence(self, frames: list) -> np.ndarray:
        """
        Extract features from a sequence of frames.

        Returns:
            Array of shape (number_of_frames, 126).
        """
        if not frames:
            return np.empty((0, 126), dtype=np.float32)

        sequence = []

        for frame in frames:
            sequence.append(self.extract(frame))

        return np.asarray(sequence, dtype=np.float32)


# Shared singleton used by the recognition routes and application lifespan.
extractor = LandmarkExtractor()
