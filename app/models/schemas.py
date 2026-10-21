"""
Pydantic schemas for all API request / response bodies.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "signspeak-ai"
    model: str = "signspeak_combined_best"
    classes: int
    ready: bool = True
    predictor_ready: bool = True
    extractor_ready: bool = True


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    predictor_ready: bool
    extractor_ready: bool


class LivenessResponse(BaseModel):
    status: str = "alive"


# ─────────────────────────────────────────────────────────────
# Classes
# ─────────────────────────────────────────────────────────────

class ClassEntry(BaseModel):
    id: int
    name: str


class ClassesResponse(BaseModel):
    classes: list[ClassEntry]


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

class TopPrediction(BaseModel):
    """A single ranked prediction entry."""
    class_name: str = Field(..., alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Full prediction result returned to the client."""
    type: str = "prediction"
    prediction: str
    confidence: float                      # percentage 0-100
    recognized: bool
    top_predictions: list[TopPrediction]
    hands_detected: int
    frames_collected: int
    timestamp: int                         # Unix ms


# ─────────────────────────────────────────────────────────────
# Status (collecting phase)
# ─────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    type: str = "status"
    status: str
    frames_collected: int
    required_frames: int


# ─────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    type: str = "error"
    code: str
    message: str


# ─────────────────────────────────────────────────────────────
# Sentence buffer (for Node.js / Flutter display)
# ─────────────────────────────────────────────────────────────

class SentenceUpdate(BaseModel):
    """Emitted when the sentence buffer changes."""
    type: str = "sentence"
    words: list[str]
    sentence: str
