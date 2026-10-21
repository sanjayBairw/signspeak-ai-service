"""
SignSpeak AI Service — WebSocket Recognition Endpoint
======================================================

WebSocket: /ws/sign

Protocol
--------
Client → Server (binary):
    Raw JPEG/PNG image bytes.

Client → Server (text / JSON):
    {
        "type": "control",
        "action": "clear_sentence" | "reset"
    }

Server → Client (JSON):
    While collecting:
        { "type": "status", "status": "collecting",
          "frames_collected": N, "required_frames": 32 }

    After 32 frames:
        {
            "type": "prediction",
            "prediction": "Hello",
            "confidence": 97.42,
            "recognized": true,
            "top_predictions": [
                { "class": "Hello",     "confidence": 97.42 },
                { "class": "Thank you", "confidence":  1.21 },
                { "class": "Goodbye",   "confidence":  0.81 }
            ],
            "hands_detected": 2,
            "frames_collected": 32,
            "timestamp": 1234567890,
            "sentence": "Hello World",
            "sentence_words": ["Hello", "World"],
            "sentence_updated": true
        }

    On error:
        { "type": "error", "code": "...", "message": "..." }

One RecognitionSession is created per connection and destroyed on disconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from app.models.predictor import predictor
from app.recognition.landmark_extractor import extractor
from app.services.recognition_service import RecognitionSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recognition"])


async def _safe_send_json(websocket: WebSocket, data: dict) -> bool:
    """
    Safely send a JSON message over the WebSocket.
    Returns True if sent successfully, False if connection is closed or closing.
    Never throws RuntimeError or WebSocketDisconnect to caller.
    """
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_text(json.dumps(data))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as exc:
        logger.debug("Failed to send WebSocket message: %s", exc)
        return False


@router.websocket("/ws/sign")
async def websocket_sign(websocket: WebSocket) -> None:
    """Real-time sign recognition WebSocket endpoint."""

    # ── Check dependency readiness BEFORE accepting session ────
    if not predictor.is_loaded or not extractor.is_loaded:
        logger.warning(
            "Rejecting /ws/sign connection: AI service dependencies not ready "
            "(predictor_ready=%s, extractor_ready=%s)",
            predictor.is_loaded,
            extractor.is_loaded,
        )
        await websocket.close(
            code=status.WS_1013_TRY_AGAIN_LATER,
            reason="AI service dependencies not ready",
        )
        return

    await websocket.accept()
    connection_id = str(uuid.uuid4())[:8]
    session: Optional[RecognitionSession] = None

    logger.info("[%s] WebSocket connected and session initialized.", connection_id)

    try:
        # ── Create per-connection session ──────────────────────────
        session = RecognitionSession(connection_id)

        # ── Send initial status ────────────────────────────────────
        if not await _safe_send_json(websocket, {
            "type": "status",
            "status": "connected",
            "frames_collected": 0,
            "required_frames": 32,
        }):
            return

        # ── Message loop ───────────────────────────────────────────
        while True:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            if "bytes" in message and message["bytes"]:
                # ── Binary frame (JPEG/PNG bytes) ──────────────────
                frame_bytes: bytes = message["bytes"]

                try:
                    # Offload CPU-intensive decoding, MediaPipe extraction, and Keras
                    # prediction to a worker thread so the main asyncio event loop remains
                    # unblocked to immediately handle WebSocket keepalive pings/pongs.
                    result = await asyncio.to_thread(session.process_frame_bytes, frame_bytes)
                    if not await _safe_send_json(websocket, result):
                        break
                except ValueError as ve:
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": str(ve),
                    }):
                        break
                except RuntimeError as re:
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "MEDIAPIPE_ERROR",
                        "message": str(re),
                    }):
                        break
                except (WebSocketDisconnect, ConnectionError):
                    break
                except Exception as exc:
                    logger.exception("[%s] Unexpected frame processing error: %s", connection_id, exc)
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "MODEL_INFERENCE_ERROR",
                        "message": "Internal server error during inference.",
                    }):
                        break

            elif "text" in message and message["text"]:
                # ── Text / control message ─────────────────────────
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": "Invalid JSON in text message.",
                    }):
                        break
                    continue

                if not isinstance(payload, dict):
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": "JSON text payload must be an object.",
                    }):
                        break
                    continue

                msg_type = payload.get("type", "")

                if msg_type == "frame":
                    # Base64-encoded frame sent as JSON text
                    b64_data = payload.get("data", "")
                    if not b64_data or not isinstance(b64_data, str):
                        if not await _safe_send_json(websocket, {
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": "No frame data in JSON message.",
                        }):
                            break
                        continue
                    try:
                        result = await asyncio.to_thread(session.process_frame_b64, b64_data)
                        if not await _safe_send_json(websocket, result):
                            break
                    except ValueError as ve:
                        if not await _safe_send_json(websocket, {
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": str(ve),
                        }):
                            break
                    except RuntimeError as re:
                        if not await _safe_send_json(websocket, {
                            "type": "error",
                            "code": "MEDIAPIPE_ERROR",
                            "message": str(re),
                        }):
                            break
                    except (WebSocketDisconnect, ConnectionError):
                        break
                    except Exception as exc:
                        logger.exception("[%s] Base64 frame error: %s", connection_id, exc)
                        if not await _safe_send_json(websocket, {
                            "type": "error",
                            "code": "MODEL_INFERENCE_ERROR",
                            "message": "Internal server error during inference.",
                        }):
                            break

                elif msg_type == "control":
                    action = payload.get("action", "")
                    if action == "clear_sentence":
                        session.clear_sentence()
                        if not await _safe_send_json(websocket, {
                            "type": "sentence",
                            "words": [],
                            "sentence": "",
                        }):
                            break
                    elif action == "reset":
                        session.reset()
                        if not await _safe_send_json(websocket, {
                            "type": "status",
                            "status": "reset",
                            "frames_collected": 0,
                            "required_frames": 32,
                        }):
                            break
                    else:
                        if not await _safe_send_json(websocket, {
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": f"Unknown control action: {action}",
                        }):
                            break
                else:
                    if not await _safe_send_json(websocket, {
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": f"Unknown message type: {msg_type}",
                    }):
                        break

    except (WebSocketDisconnect, RuntimeError):
        logger.info("[%s] WebSocket client disconnected cleanly.", connection_id)
    except Exception:
        logger.exception("[%s] Unhandled WebSocket error.", connection_id)
    finally:
        if session is not None:
            session.reset()
        logger.info("[%s] WebSocket disconnected and session cleaned up.", connection_id)
