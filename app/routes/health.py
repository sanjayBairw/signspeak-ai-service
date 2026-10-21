"""
Health and class-list REST endpoints.

GET /health  → HealthResponse
GET /classes → ClassesResponse
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.models.predictor import predictor
from app.recognition.landmark_extractor import extractor
from app.models.schemas import (
    ClassEntry,
    ClassesResponse,
    HealthResponse,
    ReadinessResponse,
    LivenessResponse,
)

router = APIRouter(tags=["info"])


@router.get("/health", response_model=HealthResponse, summary="Health and readiness check")
async def health(response: Response) -> HealthResponse:
    """
    Returns service status, model name, class count, and readiness details.
    Returns HTTP 200 when ready to serve inferences, or HTTP 503 when dependencies are uninitialized.
    """
    predictor_ready = predictor.is_loaded
    extractor_ready = extractor.is_loaded
    is_ready = predictor_ready and extractor_ready

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if is_ready else ("loading" if (predictor_ready or extractor_ready) else "unhealthy"),
        service="signspeak-ai",
        model="signspeak_combined_best",
        classes=predictor.num_classes if predictor_ready else 0,
        ready=is_ready,
        predictor_ready=predictor_ready,
        extractor_ready=extractor_ready,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(response: Response) -> ReadinessResponse:
    """
    Readiness probe for orchestrators and load balancers.
    Returns HTTP 200 when both model predictor and landmark extractor are loaded and ready.
    Returns HTTP 503 if any dependency is missing or still initializing.
    """
    predictor_ready = predictor.is_loaded
    extractor_ready = extractor.is_loaded
    is_ready = predictor_ready and extractor_ready

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        ready=is_ready,
        predictor_ready=predictor_ready,
        extractor_ready=extractor_ready,
    )


@router.get("/live", response_model=LivenessResponse, summary="Liveness probe")
async def live() -> LivenessResponse:
    """
    Liveness probe: verifies the ASGI/web server process is alive.
    Always returns HTTP 200.
    """
    return LivenessResponse(status="alive")


@router.get("/classes", response_model=ClassesResponse, summary="List all sign classes")
async def list_classes() -> ClassesResponse:
    """
    Returns the full list of 96 sign classes with their IDs.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded.",
        )

    return ClassesResponse(
        classes=[
            ClassEntry(id=i, name=name)
            for i, name in enumerate(predictor.class_names)
        ]
    )
