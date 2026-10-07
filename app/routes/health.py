"""
Health and class-list REST endpoints.

GET /health  → HealthResponse
GET /classes → ClassesResponse
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.predictor import predictor
from app.models.schemas import ClassEntry, ClassesResponse, HealthResponse

router = APIRouter(tags=["info"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """
    Returns service status, model name, and class count.
    Always responds even when the model is still loading (for load-balancer probes).
    """
    return HealthResponse(
        status="ok" if predictor.is_loaded else "loading",
        service="signspeak-ai",
        model="signspeak_combined_best",
        classes=predictor.num_classes if predictor.is_loaded else 0,
    )


@router.get("/classes", response_model=ClassesResponse, summary="List all sign classes")
async def list_classes() -> ClassesResponse:
    """
    Returns the full list of 96 sign classes with their IDs.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    return ClassesResponse(
        classes=[
            ClassEntry(id=i, name=name)
            for i, name in enumerate(predictor.class_names)
        ]
    )
