"""
SignSpeak AI Service — Keras Model Predictor
=============================================

Loads the 96-class Keras model once at startup and exposes a thread-safe
predict() method. The custom LandmarkAugmentation layer is registered so
tf.keras.models.load_model() can reconstruct the full model graph.

IMPORTANT: All inference calls use training=False so augmentation
(noise + temporal masking) is completely disabled at inference time.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Custom Keras Layer — must match training exactly
# ─────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="SignSpeak")
class LandmarkAugmentation(tf.keras.layers.Layer):
    """
    Data-augmentation layer used during training.

    At inference (training=False) it is a strict no-op — the inputs
    are returned unchanged.  This matches the training script behaviour:

        if not training:
            return inputs
    """

    def __init__(
        self,
        noise_std: float = 0.008,
        mask_probability: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.mask_probability = mask_probability

    def call(self, inputs, training=None):  # type: ignore[override]
        # At inference training is None or False — pass through unchanged.
        if training is not True:
            return inputs

        # Training-time augmentation (never reached during inference)
        noise = tf.random.normal(
            tf.shape(inputs),
            mean=0.0,
            stddev=self.noise_std,
            dtype=inputs.dtype,
        )
        x = inputs + noise

        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        random_values = tf.random.uniform([batch_size, time_steps])
        mask = tf.cast(random_values > self.mask_probability, x.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        return x * mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "noise_std": self.noise_std,
                "mask_probability": self.mask_probability,
            }
        )
        return config


# ─────────────────────────────────────────────────────────────
# Predictor singleton
# ─────────────────────────────────────────────────────────────

class SignPredictor:
    """
    Loads and holds the Keras model, normalization stats, and class
    names.  Exposes a single predict() call that is safe to call from
    an async context (it is CPU-bound but fast enough for our needs;
    wrap in run_in_executor() if needed).
    """

    def __init__(self) -> None:
        self._model: Optional[tf.keras.Model] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._class_names: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model + normalization stats + class names.  Call once at startup."""
        settings = get_settings()

        # ── Validate file existence ────────────────────────────────
        required = {
            "MODEL_PATH": settings.model_path,
            "MEAN_PATH": settings.mean_path,
            "STD_PATH": settings.std_path,
            "CLASS_PATH": settings.class_path,
        }
        for env_key, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[MODEL_NOT_FOUND] Required file missing — "
                    f"set {env_key} environment variable. "
                    f"Path checked: {path}"
                )

        # ── Model ──────────────────────────────────────────────────
        logger.info("Loading Keras model…")
        t0 = time.perf_counter()
        self._model = tf.keras.models.load_model(
            settings.model_path,
            custom_objects={"LandmarkAugmentation": LandmarkAugmentation},
            compile=False,
        )
        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.2fs — input=%s output=%s",
                    elapsed, self._model.input_shape, self._model.output_shape)

        # Sanity-check shape
        if self._model.input_shape != (None, settings.sequence_length, settings.features_per_frame):
            raise RuntimeError(
                f"[INVALID_SEQUENCE] Model input shape mismatch. "
                f"Expected (None, {settings.sequence_length}, {settings.features_per_frame}), "
                f"got {self._model.input_shape}"
            )
        if self._model.output_shape != (None, 96):
            raise RuntimeError(
                f"[INVALID_MODEL] Model output shape mismatch. "
                f"Expected (None, 96), got {self._model.output_shape}"
            )

        # ── Normalization ─────────────────────────────────────────
        logger.info("Loading normalization stats…")
        self._feature_mean = np.load(settings.mean_path).astype(np.float32).reshape(settings.features_per_frame)
        self._feature_std = np.load(settings.std_path).astype(np.float32).reshape(settings.features_per_frame)
        # Prevent division by zero (same guard used during training)
        self._feature_std = np.where(self._feature_std < 1e-6, 1.0, self._feature_std).astype(np.float32)
        logger.info("Normalization stats loaded — mean shape=%s std shape=%s",
                    self._feature_mean.shape, self._feature_std.shape)

        # ── Class names ───────────────────────────────────────────
        logger.info("Loading class names…")
        with open(settings.class_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Handle list or dict formats produced by training scripts
        if isinstance(raw, dict):
            if "class_names" in raw:
                raw = raw["class_names"]
            elif "classes" in raw:
                raw = raw["classes"]
            else:
                try:
                    raw = [raw[str(i)] for i in range(len(raw))]
                except Exception:
                    raw = list(raw.values())

        self._class_names = [str(x).strip() for x in raw]

        if len(self._class_names) != 96:
            raise RuntimeError(
                f"Expected 96 classes but loaded {len(self._class_names)}. "
                "Check CLASS_PATH environment variable."
            )
        logger.info("Class names loaded — %d classes", len(self._class_names))

        # ── Warm-up ───────────────────────────────────────────────
        logger.info("Warming up model…")
        warmup_input = np.zeros((1, settings.sequence_length, settings.features_per_frame), dtype=np.float32)
        _ = self._model(warmup_input, training=False).numpy()
        logger.info("Model warm-up complete.")

        self._loaded = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    def normalize(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply per-feature z-score normalization identical to training:
            normalized = (sequence - feature_mean) / feature_std
        Input shape:  (32, 126)
        Output shape: (32, 126)
        """
        assert self._feature_mean is not None and self._feature_std is not None
        return (sequence - self._feature_mean) / self._feature_std

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        sequence: np.ndarray,
        top_k: int = 3,
        confidence_threshold: Optional[float] = None,
    ) -> dict:
        """
        Run inference on a (32, 126) float32 sequence.

        Returns a dict that maps directly to PredictionResponse schema.

        Args:
            sequence:             Raw (un-normalized) landmark sequence (32, 126).
            top_k:                Number of top predictions to return.
            confidence_threshold: Override settings threshold if provided.
        """
        if not self._loaded:
            raise RuntimeError("[MODEL_NOT_FOUND] Predictor not loaded. Call load() first.")

        settings = get_settings()
        threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold

        if sequence.shape != (settings.sequence_length, settings.features_per_frame):
            raise ValueError(
                f"[INVALID_SEQUENCE] Expected ({settings.sequence_length}, "
                f"{settings.features_per_frame}), got {sequence.shape}"
            )

        # Normalize
        normed = self.normalize(sequence)

        # Model input: (1, 32, 126)
        model_input = normed[np.newaxis, ...].astype(np.float32)

        # Run inference — training=False ensures augmentation is disabled
        probs = self._model(model_input, training=False).numpy()[0]  # (96,)

        # Build results
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "class": self._class_names[i],
                "confidence": float(round(probs[i] * 100, 4)),
            }
            for i in top_indices
        ]

        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx] * 100)
        recognized = best_conf >= threshold
        prediction_label = self._class_names[best_idx] if recognized else "UNKNOWN"

        return {
            "prediction": prediction_label,
            "confidence": round(best_conf, 4),
            "recognized": recognized,
            "top_predictions": top_predictions,
        }


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

predictor = SignPredictor()
