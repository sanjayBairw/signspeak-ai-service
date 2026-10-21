"""
SignSpeak AI Service — Recognition Service
==========================================

Orchestrates the full frame → prediction pipeline:

    raw frame bytes / base64
         ↓
    decode (preprocessing)
         ↓
    MediaPipe landmark extraction
         ↓
    add to per-connection sequence buffer
         ↓
    (when buffer full + interval elapsed)
         ↓
    normalize + Keras model inference
         ↓
    structured prediction dict

Also manages the per-connection sentence buffer (duplicate suppression).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from app.config import get_settings
from app.models.predictor import predictor
from app.recognition.landmark_extractor import extractor
from app.recognition.preprocessing import decode_base64_frame, decode_image_bytes, validate_feature_vector
from app.recognition.sequence_manager import SequenceBuffer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Sentence buffer (per-connection)
# ─────────────────────────────────────────────────────────────

class SentenceBuffer:
    """
    Accumulates recognized words into a sentence.

    Duplicate suppression: if the same word appears consecutively,
    it is only added once until either:
      - a different word appears, OR
      - DUPLICATE_COOLDOWN seconds have elapsed
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._cooldown = settings.duplicate_cooldown
        self._words: list[str] = []
        self._last_word: Optional[str] = None
        self._last_word_time: float = 0.0

    def try_add(self, word: str) -> bool:
        """
        Attempt to add a word to the sentence.

        Returns True if the word was actually added (new to sentence).
        """
        now = time.monotonic()
        is_duplicate = (word == self._last_word)
        cooldown_elapsed = (now - self._last_word_time) >= self._cooldown

        if is_duplicate and not cooldown_elapsed:
            return False

        self._words.append(word)
        self._last_word = word
        self._last_word_time = now
        return True

    def clear(self) -> None:
        self._words.clear()
        self._last_word = None
        self._last_word_time = 0.0

    @property
    def words(self) -> list[str]:
        return list(self._words)

    @property
    def sentence(self) -> str:
        return " ".join(self._words)


# ─────────────────────────────────────────────────────────────
# Per-connection recognition session
# ─────────────────────────────────────────────────────────────

class RecognitionSession:
    """
    One per active WebSocket connection.

    Encapsulates the sequence buffer and sentence buffer for a single
    client session.  Thread safety is handled inside SequenceBuffer.
    """

    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.sequence_buffer = SequenceBuffer()
        self.sentence_buffer = SentenceBuffer()
        self._created_at = time.monotonic()

    def process_frame_bytes(self, frame_bytes: bytes) -> dict:
        """
        Process raw image bytes and return either a status or prediction dict.

        Args:
            frame_bytes: JPEG/PNG image bytes from the client.

        Returns:
            dict with type == "status"     while collecting frames, or
            dict with type == "prediction" once 32 frames are ready.
        """
        # 1. Decode
        bgr_frame = decode_image_bytes(frame_bytes)

        return self._run_pipeline(bgr_frame)

    def process_frame_b64(self, b64_data: str) -> dict:
        """
        Process a base64-encoded frame and return status or prediction dict.
        """
        bgr_frame = decode_base64_frame(b64_data)
        return self._run_pipeline(bgr_frame)

    def _run_pipeline(self, bgr_frame: np.ndarray) -> dict:
        """Internal pipeline shared by both frame input paths."""
        settings = get_settings()

        # 2. MediaPipe landmark extraction
        features, hands_detected = extractor.extract_from_bgr(bgr_frame)

        # 3. Validate feature vector
        if not validate_feature_vector(features):
            logger.error("[%s] Invalid feature vector generated — reporting error.", self.connection_id)
            raise ValueError("[INVALID_FRAME] Invalid landmark feature vector (non-finite or incorrect shape).")

        # 4. Add to sequence buffer
        self.sequence_buffer.add_frame(features, hands_detected)

        frames_collected = self.sequence_buffer.frames_collected
        required = settings.sequence_length

        # 5. Return status if not yet full
        if not self.sequence_buffer.is_ready:
            return {
                "type": "status",
                "status": "collecting",
                "frames_collected": frames_collected,
                "required_frames": required,
            }

        # 6. Rate-limit predictions
        if not self.sequence_buffer.should_predict():
            return {
                "type": "status",
                "status": "ready",
                "frames_collected": frames_collected,
                "required_frames": required,
            }

        # 7. Snapshot + predict
        sequence = self.sequence_buffer.snapshot()
        if sequence is None:
            return {
                "type": "status",
                "status": "collecting",
                "frames_collected": frames_collected,
                "required_frames": required,
            }

        self.sequence_buffer.mark_predicted()

        result = predictor.predict(sequence)

        # 8. Update sentence buffer
        recognized = result.get("recognized", False)
        prediction_label = result.get("prediction", "UNKNOWN")
        sentence_updated = False
        if recognized and prediction_label != "UNKNOWN":
            sentence_updated = self.sentence_buffer.try_add(prediction_label)

        return {
            "type": "prediction",
            "prediction": prediction_label,
            "confidence": result["confidence"],
            "recognized": recognized,
            "top_predictions": result["top_predictions"],
            "hands_detected": self.sequence_buffer.last_hands_detected,
            "frames_collected": frames_collected,
            "timestamp": int(time.time() * 1000),
            "sentence": self.sentence_buffer.sentence,
            "sentence_words": self.sentence_buffer.words,
            "sentence_updated": sentence_updated,
        }

    def clear_sentence(self) -> None:
        """Reset the sentence buffer (e.g. on user request)."""
        self.sentence_buffer.clear()

    def reset(self) -> None:
        """Full reset: sequence buffer + sentence buffer."""
        self.sequence_buffer.clear()
        self.sentence_buffer.clear()
