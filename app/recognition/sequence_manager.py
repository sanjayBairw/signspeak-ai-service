"""
SignSpeak AI Service — Per-Connection Sequence Buffer
=====================================================

Maintains a rolling deque of (32) landmark frames per WebSocket connection.
Uses a bounded deque so the buffer never grows indefinitely.

Design decisions matching the spec:
- deque(maxlen=32) — bounded rolling sequence
- Tracks 'hands_detected' for the most recent frame
- Exposes a snapshot() method that returns a (32, 126) ndarray
- Thread-safe via a simple lock (each connection is its own object)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from app.config import get_settings


class SequenceBuffer:
    """
    Per-WebSocket-connection frame buffer.

    Each call to add_frame() pushes a (126,) feature vector.
    Once the buffer has >= sequence_length frames, snapshot() returns
    a (32, 126) ndarray suitable for model inference.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._seq_len = settings.sequence_length
        self._features = settings.features_per_frame
        self._buffer: deque[np.ndarray] = deque(maxlen=self._seq_len)
        self._lock = threading.Lock()
        self._last_hands_detected: int = 0
        # Timestamp of the last prediction (for rate-limiting)
        self._last_prediction_time: float = 0.0

    # ------------------------------------------------------------------
    # Frame management
    # ------------------------------------------------------------------

    def add_frame(self, features: np.ndarray, hands_detected: int) -> None:
        """
        Push a (126,) feature vector into the buffer.

        Args:
            features:       Feature vector from LandmarkExtractor.
            hands_detected: How many hands were seen in this frame.
        """
        if features.shape != (self._features,):
            raise ValueError(
                f"[INVALID_SEQUENCE] Expected feature shape ({self._features},), "
                f"got {features.shape}"
            )
        with self._lock:
            self._buffer.append(features.astype(np.float32))
            self._last_hands_detected = hands_detected

    def clear(self) -> None:
        """Reset the buffer (e.g. on disconnect or sign boundary)."""
        with self._lock:
            self._buffer.clear()
            self._last_hands_detected = 0
            self._last_prediction_time = 0.0

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def frames_collected(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when we have a full 32-frame sequence."""
        with self._lock:
            return len(self._buffer) >= self._seq_len

    @property
    def last_hands_detected(self) -> int:
        with self._lock:
            return self._last_hands_detected

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Optional[np.ndarray]:
        """
        Return a (32, 126) float32 copy of the current buffer, or None
        if the buffer has fewer than 32 frames.

        The copy is independent so the calling code can normalize/mutate
        it without affecting the live buffer.
        """
        with self._lock:
            if len(self._buffer) < self._seq_len:
                return None
            # deque is ordered oldest→newest; take last seq_len frames
            frames = list(self._buffer)[-self._seq_len:]
            return np.array(frames, dtype=np.float32)

    # ------------------------------------------------------------------
    # Prediction rate-limiting
    # ------------------------------------------------------------------

    def should_predict(self) -> bool:
        """
        Returns True if enough time has passed since the last prediction
        to run inference again (based on PREDICTION_INTERVAL setting).
        """
        settings = get_settings()
        now = time.monotonic()
        with self._lock:
            return (now - self._last_prediction_time) >= settings.prediction_interval

    def mark_predicted(self) -> None:
        """Call after running a prediction to reset the rate-limit timer."""
        with self._lock:
            self._last_prediction_time = time.monotonic()
