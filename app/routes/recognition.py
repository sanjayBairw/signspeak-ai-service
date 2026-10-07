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

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.predictor import predictor
from app.services.recognition_service import RecognitionSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recognition"])


@router.websocket("/ws/sign")
async def websocket_sign(websocket: WebSocket) -> None:
    """Real-time sign recognition WebSocket endpoint."""

    await websocket.accept()
    connection_id = str(uuid.uuid4())[:8]
    session: Optional[RecognitionSession] = None

    logger.info("[%s] WebSocket connected.", connection_id)

    try:
        # ── Wait until model is ready ──────────────────────────────
        if not predictor.is_loaded:
            await websocket.send_text(json.dumps({
                "type": "error",
                "code": "MODEL_NOT_FOUND",
                "message": "AI model is still loading. Please retry in a moment.",
            }))
            await websocket.close()
            return

        # ── Create per-connection session ──────────────────────────
        session = RecognitionSession(connection_id)

        # ── Send initial status ────────────────────────────────────
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "connected",
            "frames_collected": 0,
            "required_frames": 32,
        }))

        # ── Message loop ───────────────────────────────────────────
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in message and message["bytes"]:
                # ── Binary frame (JPEG/PNG bytes) ──────────────────
                frame_bytes: bytes = message["bytes"]

                try:
                    result = session.process_frame_bytes(frame_bytes)
                    await websocket.send_text(json.dumps(result))
                except ValueError as ve:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": str(ve),
                    }))
                except RuntimeError as re:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "MEDIAPIPE_ERROR",
                        "message": str(re),
                    }))
                except Exception as exc:
                    logger.exception("[%s] Unexpected frame processing error.", connection_id)
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "MODEL_INFERENCE_ERROR",
                        "message": "Internal server error during inference.",
                    }))

            elif "text" in message and message["text"]:
                # ── Text / control message ─────────────────────────
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": "Invalid JSON in text message.",
                    }))
                    continue

                msg_type = payload.get("type", "")

                if msg_type == "frame":
                    # Base64-encoded frame sent as JSON text
                    b64_data = payload.get("data", "")
                    if not b64_data:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": "No frame data in JSON message.",
                        }))
                        continue
                    try:
                        result = session.process_frame_b64(b64_data)
                        await websocket.send_text(json.dumps(result))
                    except ValueError as ve:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": str(ve),
                        }))
                    except RuntimeError as re:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "MEDIAPIPE_ERROR",
                            "message": str(re),
                        }))
                    except Exception:
                        logger.exception("[%s] Base64 frame error.", connection_id)
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "MODEL_INFERENCE_ERROR",
                            "message": "Internal server error during inference.",
                        }))

                elif msg_type == "control":
                    action = payload.get("action", "")
                    if action == "clear_sentence":
                        session.clear_sentence()
                        await websocket.send_text(json.dumps({
                            "type": "sentence",
                            "words": [],
                            "sentence": "",
                        }))
                    elif action == "reset":
                        session.reset()
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "status": "reset",
                            "frames_collected": 0,
                            "required_frames": 32,
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "code": "INVALID_FRAME",
                            "message": f"Unknown control action: {action}",
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "code": "INVALID_FRAME",
                        "message": f"Unknown message type: {msg_type}",
                    }))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] Unhandled WebSocket error.", connection_id)
    finally:
        logger.info("[%s] WebSocket disconnected.", connection_id)
