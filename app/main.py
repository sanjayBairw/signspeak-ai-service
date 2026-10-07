"""
SignSpeak AI Service — FastAPI Application
==========================================

Application factory: creates the FastAPI app, registers startup/shutdown
lifecycle hooks, mounts CORS, and includes all routers.

Model and MediaPipe are loaded ONCE during the lifespan startup event.
They are NEVER reloaded per-request.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.predictor import predictor
from app.recognition.landmark_extractor import extractor
from app.routes.health import router as health_router
from app.routes.recognition import router as recognition_router

# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ─────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: configure logging, log environment info, load model and MediaPipe.
    On shutdown: (cleanup if needed).
    """
    settings = get_settings()
    _setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("SignSpeak AI Service starting up")
    logger.info("=" * 60)
    logger.info("Python:         %s", sys.version.split()[0])

    # Log TF version safely
    try:
        import tensorflow as tf
        logger.info("TensorFlow:     %s", tf.__version__)
    except ImportError:
        logger.error("TensorFlow is NOT installed. Please install it.")

    # Log MediaPipe version safely
    try:
        import mediapipe as mp
        logger.info("MediaPipe:      %s", mp.__version__)
    except ImportError:
        logger.error("MediaPipe is NOT installed. Please install it.")

    logger.info("Service:        %s v%s", settings.service_name, settings.service_version)
    logger.info("Host:           %s:%d", settings.host, settings.port)
    logger.info("Confidence thr: %.1f%%", settings.confidence_threshold)
    logger.info("Sequence len:   %d frames", settings.sequence_length)
    logger.info("Prediction intv: %.3fs", settings.prediction_interval)

    # ── Load model ────────────────────────────────────────────
    try:
        predictor.load()
        logger.info("Model loaded: 96 classes, input=(None, 32, 126)")
    except FileNotFoundError as exc:
        logger.error("CRITICAL: %s", exc)
        logger.error("AI service will start but /ws/sign will reject connections.")
    except Exception as exc:
        logger.exception("CRITICAL: Model load failed — %s", exc)

    # ── Load MediaPipe ────────────────────────────────────────
    try:
        extractor.load()
        logger.info("MediaPipe HandLandmarker ready.")
    except FileNotFoundError as exc:
        logger.error("CRITICAL: %s", exc)
    except Exception as exc:
        logger.exception("CRITICAL: MediaPipe load failed — %s", exc)

    logger.info("=" * 60)
    logger.info("Startup complete. Service ready.")
    logger.info("=" * 60)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("SignSpeak AI Service shutting down.")


# ─────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="SignSpeak AI Service",
        description=(
            "Production-ready Python AI service for real-time ISL sign recognition. "
            "Uses MediaPipe HandLandmarker + 96-class Keras model."
        ),
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────
    origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(recognition_router)

    return app


app = create_app()
