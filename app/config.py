"""
SignSpeak AI Service — Configuration
=====================================
All settings are loaded from environment variables (or .env file).
No Windows absolute paths are ever returned in API responses.
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings, populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
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
    # Configurable via environment variables. If not provided or invalid,
    # auto-resolves to repository models/ & assets/ directories.
    model_path: str = Field("", alias="MODEL_PATH")
    mean_path: str = Field("", alias="MEAN_PATH")
    std_path: str = Field("", alias="STD_PATH")
    class_path: str = Field("", alias="CLASS_PATH")
    hand_model_path: str = Field("", alias="HAND_MODEL_PATH")

    def model_post_init(self, __context) -> None:
        """Resolve model artifacts across env paths, repository dirs, and fallback archives."""
        ai_root = Path(__file__).resolve().parent.parent

        self.model_path = self._resolve_path(
            self.model_path,
            ai_root / "models" / "signspeak_combined_best.keras",
            r"D:\Downloads\archive\processed\models_combined\signspeak_combined_best.keras",
        )
        self.mean_path = self._resolve_path(
            self.mean_path,
            ai_root / "models" / "feature_mean.npy",
            r"D:\Downloads\archive\processed\models_combined\feature_mean.npy",
        )
        self.std_path = self._resolve_path(
            self.std_path,
            ai_root / "models" / "feature_std.npy",
            r"D:\Downloads\archive\processed\models_combined\feature_std.npy",
        )
        self.class_path = self._resolve_path(
            self.class_path,
            ai_root / "models" / "class_names.json",
            r"D:\Downloads\archive\processed\models_combined\class_names.json",
        )
        self.hand_model_path = self._resolve_path(
            self.hand_model_path,
            ai_root / "assets" / "hand_landmarker.task",
            r"D:\Downloads\archive\hand_landmarker.task",
        )

    @staticmethod
    def _resolve_path(env_val: str, rel_path: Path, archive_path: str) -> str:
        """
        Helper to resolve path:
        1. If env_val is explicitly set: use it (as absolute/cwd or relative to ai_root).
           If not found, return env_val so startup validation produces a clear FileNotFoundError.
        2. If env_val is not set: use default repository path (rel_path) or fallback archive.
        """
        ai_root = rel_path.parent.parent
        if env_val:
            if os.path.exists(env_val):
                return str(Path(env_val).resolve())
            rel_to_root = ai_root / env_val
            if rel_to_root.exists():
                return str(rel_to_root.resolve())
            # Explicit override was specified but not found — do not silently fall back
            return env_val

        # No env var specified: resolve to repository assets
        if rel_path.exists():
            return str(rel_path.resolve())
        if archive_path and os.path.exists(archive_path):
            return str(Path(archive_path).resolve())
        return str(rel_path)

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
