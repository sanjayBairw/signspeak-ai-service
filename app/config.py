"""
SignSpeak AI Service — Configuration
=====================================
All settings are loaded from environment variables (or .env file).
No Windows absolute paths are ever returned in API responses.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings, populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service ───────────────────────────────────────────────────
    service_name: str = Field("signspeak-ai", alias="SERVICE_NAME")
    service_version: str = Field("1.0.0", alias="SERVICE_VERSION")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")
    debug: bool = Field(False, alias="DEBUG")

    # ── CORS ──────────────────────────────────────────────────────
    cors_origins: str = Field("*", alias="CORS_ORIGINS")

    # ── Model Files ───────────────────────────────────────────────
    model_path: str = Field(
        r"D:\Downloads\archive\processed\models_combined\signspeak_combined_best.keras",
        alias="MODEL_PATH",
    )
    mean_path: str = Field(
        r"D:\Downloads\archive\processed\models_combined\feature_mean.npy",
        alias="MEAN_PATH",
    )
    std_path: str = Field(
        r"D:\Downloads\archive\processed\models_combined\feature_std.npy",
        alias="STD_PATH",
    )
    class_path: str = Field(
        r"D:\Downloads\archive\processed\models_combined\class_names.json",
        alias="CLASS_PATH",
    )
    hand_model_path: str = Field(
        r"D:\Downloads\archive\hand_landmarker.task",
        alias="HAND_MODEL_PATH",
    )

    # ── Recognition ───────────────────────────────────────────────
    sequence_length: int = Field(32, alias="SEQUENCE_LENGTH")
    features_per_frame: int = Field(126, alias="FEATURES_PER_FRAME")
    num_hands: int = Field(2, alias="NUM_HANDS")
    inference_width: int = Field(384, alias="INFERENCE_WIDTH")

    # ── Thresholds ────────────────────────────────────────────────
    confidence_threshold: float = Field(60.0, alias="CONFIDENCE_THRESHOLD")

    # MediaPipe confidence settings (lowered to match realtime_test.py for blur tolerance)
    min_hand_detection_confidence: float = Field(0.20, alias="MIN_HAND_DETECTION_CONFIDENCE")
    min_hand_presence_confidence: float = Field(0.20, alias="MIN_HAND_PRESENCE_CONFIDENCE")
    min_tracking_confidence: float = Field(0.20, alias="MIN_TRACKING_CONFIDENCE")

    # ── Realtime Performance ──────────────────────────────────────
    # Minimum interval (seconds) between consecutive predictions per connection.
    prediction_interval: float = Field(0.08, alias="PREDICTION_INTERVAL")

    # Max WebSocket frame size in bytes (5 MB per frame)
    max_frame_size_bytes: int = Field(5 * 1024 * 1024, alias="MAX_FRAME_SIZE_BYTES")

    # ── Sentence Builder / Duplicate Suppression ─────────────────
    duplicate_cooldown: float = Field(2.0, alias="DUPLICATE_COOLDOWN")

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── Node.js Backend URL (for future integration) ──────────────
    node_backend_url: str = Field("http://127.0.0.1:5000", alias="NODE_BACKEND_URL")

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS as a list (split on comma)."""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
