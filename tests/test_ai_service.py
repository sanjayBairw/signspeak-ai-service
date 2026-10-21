"""
SignSpeak AI Service — Test Suite
==================================

Tests:
 1. Model loading
 2. Class loading (96 classes)
 3. Landmark extraction — feature shape == (126,)
 4. Sequence shape == (32, 126)
 5. Normalization
 6. Full prediction pipeline
 7. Top-3 predictions
 8. Unknown threshold (low confidence → UNKNOWN)
 9. WebSocket connection + message exchange
10. Invalid frame handling
11. Known-video sanity test (Hello video → Hello in top predictions)

Run with:
    cd D:\\signspeak\\signspeak-main\\ai_service
    pytest tests/ -v
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# Ensure ai_service root is on the path so imports work
# regardless of how pytest is invoked.
# ─────────────────────────────────────────────────────────────

AI_SERVICE_ROOT = Path(__file__).parent.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

# ─────────────────────────────────────────────────────────────
# Dataset / model paths (resolves env var -> local repo models -> archive fallback)
# ─────────────────────────────────────────────────────────────

BASE = r"D:\Downloads\archive"

def _resolve(env_var: str, rel_subpath: str, archive_subpath: str) -> str:
    env_p = os.environ.get(env_var, "")
    if env_p and os.path.exists(env_p):
        return env_p
    local_p = os.path.join(AI_SERVICE_ROOT, rel_subpath)
    if os.path.exists(local_p):
        return local_p
    archive_p = os.path.join(BASE, archive_subpath)
    if os.path.exists(archive_p):
        return archive_p
    return local_p

MODEL_PATH = _resolve("MODEL_PATH", "models/signspeak_combined_best.keras", "processed/models_combined/signspeak_combined_best.keras")
MEAN_PATH  = _resolve("MEAN_PATH", "models/feature_mean.npy", "processed/models_combined/feature_mean.npy")
STD_PATH   = _resolve("STD_PATH", "models/feature_std.npy", "processed/models_combined/feature_std.npy")
CLASS_PATH = _resolve("CLASS_PATH", "models/class_names.json", "processed/models_combined/class_names.json")
HAND_MODEL_PATH = _resolve("HAND_MODEL_PATH", "assets/hand_landmarker.task", "hand_landmarker.task")

# Known "Hello" video for the sanity test (pick first available)
HELLO_VIDEO_CANDIDATES = [
    os.path.join(BASE, "Sample Videos", "Hello.mp4"),
    os.path.join(BASE, "Video_Dataset", "Video_Dataset", "Hello", "WIN_20231103_13_22_45_Pro.mp4"),
    os.path.join(BASE, "Video_Dataset", "Video_Dataset", "Hello", "Hello_1.mp4"),
    os.path.join(BASE, "Video_Dataset", "Video_Dataset", "Hello", "Hello_2.mp4"),
]
HELLO_VIDEO = next((p for p in HELLO_VIDEO_CANDIDATES if os.path.exists(p)), None)

# ─────────────────────────────────────────────────────────────
# Skip all tests gracefully when model files are missing
# (e.g. CI environment without the dataset)
# ─────────────────────────────────────────────────────────────

MODEL_FILES_AVAILABLE = all(os.path.exists(p) for p in [
    MODEL_PATH, MEAN_PATH, STD_PATH, CLASS_PATH, HAND_MODEL_PATH
])

requires_model = pytest.mark.skipif(
    not MODEL_FILES_AVAILABLE,
    reason="Model files not available (set MODEL_PATH etc.)"
)

requires_hello_video = pytest.mark.skipif(
    HELLO_VIDEO is None,
    reason="No Hello video found in dataset"
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def loaded_predictor():
    """Load the Keras predictor once for the whole test session."""
    import os
    os.environ["MODEL_PATH"] = MODEL_PATH
    os.environ["MEAN_PATH"]  = MEAN_PATH
    os.environ["STD_PATH"]   = STD_PATH
    os.environ["CLASS_PATH"] = CLASS_PATH
    os.environ["HAND_MODEL_PATH"] = HAND_MODEL_PATH

    # Invalidate cached settings so env vars are picked up
    from app.config import get_settings
    get_settings.cache_clear()

    from app.models.predictor import predictor
    if not predictor.is_loaded:
        predictor.load()
    return predictor


@pytest.fixture(scope="session")
def loaded_extractor():
    """Load the MediaPipe extractor once for the whole test session."""
    from app.recognition.landmark_extractor import extractor
    if not extractor.is_loaded:
        extractor.load()
    return extractor


@pytest.fixture(scope="session")
def class_names(loaded_predictor):
    return loaded_predictor.class_names


# ─────────────────────────────────────────────────────────────
# 1. Model loading & Dimensions
# ─────────────────────────────────────────────────────────────

@requires_model
def test_model_loading(loaded_predictor):
    assert loaded_predictor.is_loaded, "Predictor should be loaded"


@requires_model
def test_model_dimensions(loaded_predictor):
    """Verify input shape (None, 32, 126) and output shape (None, 96)."""
    assert loaded_predictor._model.input_shape == (None, 32, 126)
    assert loaded_predictor._model.output_shape == (None, 96)


# ─────────────────────────────────────────────────────────────
# 2. Class loading
# ─────────────────────────────────────────────────────────────

@requires_model
def test_class_count(loaded_predictor):
    assert loaded_predictor.num_classes == 96, (
        f"Expected 96 classes, got {loaded_predictor.num_classes}"
    )


@requires_model
def test_class_names_are_strings(loaded_predictor):
    for name in loaded_predictor.class_names:
        assert isinstance(name, str) and len(name) > 0, f"Invalid class name: {name!r}"


# ─────────────────────────────────────────────────────────────
# 3. Landmark extraction — feature shape == (126,)
# ─────────────────────────────────────────────────────────────

@requires_model
def test_landmark_extraction_shape_no_hand(loaded_extractor):
    """A blank white image should produce a (126,) all-zeros feature vector."""
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    features, hands_detected = loaded_extractor.extract_from_bgr(blank)
    assert features.shape == (126,), f"Feature shape wrong: {features.shape}"
    assert hands_detected == 0

    # Also verify direct extract() method returns identical features
    features_direct = loaded_extractor.extract(blank)
    assert features_direct.shape == (126,)
    assert np.allclose(features, features_direct)


# ─────────────────────────────────────────────────────────────
# 4. Sequence shape == (32, 126)
# ─────────────────────────────────────────────────────────────

@requires_model
def test_sequence_shape():
    from app.recognition.sequence_manager import SequenceBuffer
    import os
    os.environ["SEQUENCE_LENGTH"] = "32"
    os.environ["FEATURES_PER_FRAME"] = "126"

    from app.config import get_settings
    get_settings.cache_clear()

    buf = SequenceBuffer()
    for _ in range(32):
        buf.add_frame(np.zeros(126, dtype=np.float32), 0)

    snap = buf.snapshot()
    assert snap is not None, "Snapshot should not be None after 32 frames"
    assert snap.shape == (32, 126), f"Sequence shape wrong: {snap.shape}"


# ─────────────────────────────────────────────────────────────
# 5. Normalization
# ─────────────────────────────────────────────────────────────

@requires_model
def test_normalization(loaded_predictor):
    """Normalization of all-zero input should give -mean/std (non-zero where mean≠0)."""
    zeros = np.zeros((32, 126), dtype=np.float32)
    normed = loaded_predictor.normalize(zeros)
    assert normed.shape == (32, 126), "Normalized shape wrong"
    assert np.isfinite(normed).all(), "Normalization produced NaN/Inf"


# ─────────────────────────────────────────────────────────────
# 6. Prediction runs without error
# ─────────────────────────────────────────────────────────────

@requires_model
def test_prediction_runs(loaded_predictor):
    """Model inference on zeros should complete without exception."""
    sequence = np.zeros((32, 126), dtype=np.float32)
    result = loaded_predictor.predict(sequence)
    assert "prediction" in result
    assert "confidence" in result
    assert "recognized" in result
    assert "top_predictions" in result
    assert isinstance(result["confidence"], float)


# ─────────────────────────────────────────────────────────────
# 7. Top-3 predictions
# ─────────────────────────────────────────────────────────────

@requires_model
def test_top3_predictions(loaded_predictor):
    sequence = np.zeros((32, 126), dtype=np.float32)
    result = loaded_predictor.predict(sequence, top_k=3)
    tops = result["top_predictions"]
    assert len(tops) == 3, f"Expected 3 top predictions, got {len(tops)}"
    for entry in tops:
        assert "class" in entry
        assert "confidence" in entry
        assert 0.0 <= entry["confidence"] <= 100.0


# ─────────────────────────────────────────────────────────────
# 8. Unknown threshold
# ─────────────────────────────────────────────────────────────

@requires_model
def test_unknown_threshold(loaded_predictor):
    """
    When best confidence < threshold (forced to 100%), result should be UNKNOWN.
    We test by setting a very high threshold (100%) so any real output is UNKNOWN.
    """
    sequence = np.zeros((32, 126), dtype=np.float32)
    result = loaded_predictor.predict(sequence, confidence_threshold=100.0)
    assert result["prediction"] == "UNKNOWN", (
        f"Expected UNKNOWN with threshold=100%, got: {result['prediction']}"
    )
    assert result["recognized"] is False


# ─────────────────────────────────────────────────────────────
# 9. WebSocket connection test
# ─────────────────────────────────────────────────────────────

@requires_model
@pytest.mark.asyncio
async def test_websocket_connection():
    """
    Test that the WebSocket endpoint accepts a connection and sends
    the initial connected status message.
    """
    import asyncio
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            msg = ws.receive_json()
            # Should get either a connected status or model-not-loaded error
            assert msg["type"] in ("status", "error"), f"Unexpected first message: {msg}"
            if msg["type"] == "status":
                assert msg["status"] == "connected"


# ─────────────────────────────────────────────────────────────
# 10. Invalid frame handling
# ─────────────────────────────────────────────────────────────

@requires_model
@pytest.mark.asyncio
async def test_invalid_frame_handling():
    """Sending garbage bytes should return an error, not crash the server."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            # Consume the initial status
            _ = ws.receive_json()

            # Send garbage binary
            ws.send_bytes(b"this is not an image")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "code" in msg


@requires_model
@pytest.mark.asyncio
async def test_websocket_control_messages():
    """Test clear_sentence and reset control actions over WebSocket."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            # Consume the initial status
            init_msg = ws.receive_json()
            assert init_msg["type"] == "status"
            assert init_msg["status"] == "connected"

            # Clear sentence control
            ws.send_text(json.dumps({"type": "control", "action": "clear_sentence"}))
            clear_msg = ws.receive_json()
            assert clear_msg["type"] == "sentence"
            assert clear_msg["sentence"] == ""
            assert clear_msg["words"] == []

            # Reset control
            ws.send_text(json.dumps({"type": "control", "action": "reset"}))
            reset_msg = ws.receive_json()
            assert reset_msg["type"] == "status"
            assert reset_msg["status"] == "reset"
            assert reset_msg["frames_collected"] == 0


# ─────────────────────────────────────────────────────────────
# 12. Known video sanity test — Hello should be in top predictions
# ─────────────────────────────────────────────────────────────

@requires_model
@requires_hello_video
def test_hello_video_prediction(loaded_predictor, loaded_extractor):
    """
    Process a known 'Hello' video frame-by-frame and verify that
    'Hello' appears in the top-3 predictions.

    NOTE: This test verifies the full inference pipeline.
    It does NOT guarantee real-world accuracy for all videos.
    """
    import cv2

    sequence_length = 32
    cap = cv2.VideoCapture(HELLO_VIDEO)
    assert cap.isOpened(), f"Cannot open Hello video: {HELLO_VIDEO}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(np.linspace(0, max(total_frames - 1, 0), sequence_length).astype(int))

    frames: list[np.ndarray] = []
    frame_idx = 0

    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            frames.append(frame)
        frame_idx += 1

    cap.release()

    # Pad if video is short
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy() if frames else np.zeros((480, 640, 3), dtype=np.uint8))

    # Extract features
    feature_list = []
    for frame in frames[:sequence_length]:
        feats, _ = loaded_extractor.extract_from_bgr(frame)
        feature_list.append(feats)

    sequence = np.array(feature_list[:sequence_length], dtype=np.float32)
    assert sequence.shape == (32, 126), f"Sequence shape wrong: {sequence.shape}"

    result = loaded_predictor.predict(sequence, top_k=3, confidence_threshold=0.0)
    top_names = [p["class"].lower() for p in result["top_predictions"]]

    print(f"\nHello video result: {result['prediction']} ({result['confidence']:.2f}%)")
    print("Top-3:", result["top_predictions"])

    assert "hello" in top_names, (
        f"Expected 'Hello' in top-3 predictions, got: {result['top_predictions']}"
    )


# ─────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────

def test_decode_base64_frame():
    """A valid JPEG base64 string should decode to a BGR ndarray."""
    import cv2
    from app.recognition.preprocessing import decode_base64_frame

    # Create a tiny test image
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    frame = decode_base64_frame(b64)
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_decode_base64_frame_with_prefix():
    """Data-URI prefix should be stripped before decoding."""
    import cv2
    from app.recognition.preprocessing import decode_base64_frame

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    frame = decode_base64_frame(b64)
    assert frame is not None


def test_decode_invalid_bytes():
    """Garbage bytes should raise ValueError with INVALID_IMAGE code."""
    from app.recognition.preprocessing import decode_image_bytes
    with pytest.raises(ValueError, match="INVALID_IMAGE"):
        decode_image_bytes(b"not an image")


def test_validate_feature_vector_valid():
    from app.recognition.preprocessing import validate_feature_vector
    assert validate_feature_vector(np.zeros(126, dtype=np.float32)) is True


def test_validate_feature_vector_wrong_shape():
    from app.recognition.preprocessing import validate_feature_vector
    assert validate_feature_vector(np.zeros(100, dtype=np.float32)) is False


def test_validate_feature_vector_nan():
    from app.recognition.preprocessing import validate_feature_vector
    v = np.zeros(126, dtype=np.float32)
    v[0] = float("nan")
    assert validate_feature_vector(v) is False


def test_validate_feature_vector_inf():
    from app.recognition.preprocessing import validate_feature_vector
    v = np.zeros(126, dtype=np.float32)
    v[5] = float("inf")
    assert validate_feature_vector(v) is False


# ─────────────────────────────────────────────────────────────
# Phase 2 & 3 Focused Tests
# ─────────────────────────────────────────────────────────────

def test_artifact_path_resolution():
    """Verify that Settings._resolve_path resolves local repo artifacts and environment overrides."""
    from app.config import Settings
    from pathlib import Path
    ai_root = Path(__file__).resolve().parent.parent

    # 1. Existing file resolution
    rel_model = ai_root / "models" / "signspeak_combined_best.keras"
    resolved = Settings._resolve_path("", rel_model, "")
    assert os.path.exists(resolved)
    assert os.path.samefile(resolved, rel_model)

    # 2. Env override resolution (relative to ai_root)
    resolved_env = Settings._resolve_path("models/signspeak_combined_best.keras", rel_model, "")
    assert os.path.exists(resolved_env)
    assert os.path.samefile(resolved_env, rel_model)

    # 3. Nonexistent fallback returns candidate path
    nonexistent = Settings._resolve_path("nonexistent_model.keras", ai_root / "models" / "nonexistent.keras", "")
    assert "nonexistent" in nonexistent


def test_missing_model_file_raises_error(monkeypatch):
    """Verify that SignPredictor.load raises FileNotFoundError if model is missing."""
    from app.models.predictor import SignPredictor
    from app.config import get_settings

    p = SignPredictor()
    monkeypatch.setenv("MODEL_PATH", "nonexistent/path/model.keras")
    get_settings.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="MODEL_NOT_FOUND"):
            p.load()
    finally:
        get_settings.cache_clear()


def test_missing_asset_file_raises_error(monkeypatch):
    """Verify that LandmarkExtractor.load raises FileNotFoundError if task file is missing."""
    from app.recognition.landmark_extractor import LandmarkExtractor
    from app.config import get_settings

    monkeypatch.setenv("HAND_MODEL_PATH", "nonexistent/path/hand_landmarker.task")
    get_settings.cache_clear()
    e = LandmarkExtractor()
    try:
        with pytest.raises(FileNotFoundError):
            e.load()
    finally:
        get_settings.cache_clear()


def test_landmark_extraction_invalid_frames(loaded_extractor):
    """Corrupted/invalid frames must raise ValueError, not return all zeros."""
    # 1. None
    with pytest.raises(ValueError, match="valid NumPy image"):
        loaded_extractor.extract_from_bgr(None)

    # 2. Empty array
    with pytest.raises(ValueError, match="empty"):
        loaded_extractor.extract_from_bgr(np.zeros((0,), dtype=np.uint8))

    # 3. 2D array instead of 3D BGR
    with pytest.raises(ValueError, match="three-channel BGR image"):
        loaded_extractor.extract_from_bgr(np.zeros((100, 100), dtype=np.uint8))

    # 4. Zero dimensions / empty frame
    with pytest.raises(ValueError, match="Input frame is empty"):
        loaded_extractor.extract_from_bgr(np.zeros((0, 100, 3), dtype=np.uint8))


def test_prediction_pipeline_compatibility(loaded_predictor, loaded_extractor):
    """
    Test full frame ingestion through RecognitionSession:
    - 31 frames return status='collecting'
    - 32nd frame returns prediction structure with confidence and top predictions
    """
    import cv2
    from app.services.recognition_service import RecognitionSession

    session = RecognitionSession(connection_id="test_sess")
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", blank)
    jpeg_bytes = buf.tobytes()

    # Ingest 31 frames
    for i in range(31):
        res = session.process_frame_bytes(jpeg_bytes)
        assert res["type"] == "status"
        assert res["status"] == "collecting"
        assert res["frames_collected"] == i + 1

    # 32nd frame should produce a prediction
    res32 = session.process_frame_bytes(jpeg_bytes)
    assert res32["type"] == "prediction"
    assert "prediction" in res32
    assert "confidence" in res32
    assert "top_predictions" in res32
    assert "hands_detected" in res32
    assert len(res32["top_predictions"]) == 3


# ─────────────────────────────────────────────────────────────
# Phase 4 Focused Tests: Readiness, Health Probes & WebSocket API
# ─────────────────────────────────────────────────────────────

@requires_model
def test_health_and_readiness_when_ready(loaded_predictor, loaded_extractor):
    """Verify /health, /ready, /live, and /classes return 200 and expected payloads when service is ready."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        # GET /health
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        data = h_resp.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["predictor_ready"] is True
        assert data["extractor_ready"] is True
        assert data["classes"] == 96
        assert data["model"] == "signspeak_combined_best"
        assert data["service"] == "signspeak-ai"

        # GET /ready
        r_resp = client.get("/ready")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["status"] == "ready"
        assert r_data["ready"] is True
        assert r_data["predictor_ready"] is True
        assert r_data["extractor_ready"] is True

        # GET /live
        l_resp = client.get("/live")
        assert l_resp.status_code == 200
        assert l_resp.json()["status"] == "alive"

        # GET /classes
        c_resp = client.get("/classes")
        assert c_resp.status_code == 200
        classes = c_resp.json()["classes"]
        assert len(classes) == 96
        assert classes[0]["id"] == 0
        assert isinstance(classes[0]["name"], str)


def test_health_and_readiness_when_predictor_unavailable(loaded_extractor, monkeypatch):
    """Verify /health and /ready return 503 when predictor is not loaded."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.models.predictor import predictor

    with TestClient(app) as client:
        monkeypatch.setattr(predictor, "_loaded", False)

        h_resp = client.get("/health")
        assert h_resp.status_code == 503
        data = h_resp.json()
        assert data["ready"] is False
        assert data["predictor_ready"] is False
        assert data["extractor_ready"] is True
        assert data["classes"] == 0

        r_resp = client.get("/ready")
        assert r_resp.status_code == 503
        assert r_resp.json()["ready"] is False
        assert r_resp.json()["status"] == "not_ready"

        # Live probe still returns 200
        l_resp = client.get("/live")
        assert l_resp.status_code == 200
        assert l_resp.json()["status"] == "alive"

        # Classes returns 503
        c_resp = client.get("/classes")
        assert c_resp.status_code == 503


def test_health_and_readiness_when_extractor_unavailable(loaded_predictor, monkeypatch):
    """Verify /health and /ready return 503 when MediaPipe extractor is not loaded."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.recognition.landmark_extractor import extractor

    with TestClient(app) as client:
        monkeypatch.setattr(extractor, "_loaded", False)

        h_resp = client.get("/health")
        assert h_resp.status_code == 503
        data = h_resp.json()
        assert data["ready"] is False
        assert data["predictor_ready"] is True
        assert data["extractor_ready"] is False

        r_resp = client.get("/ready")
        assert r_resp.status_code == 503
        assert r_resp.json()["ready"] is False


def test_websocket_rejection_when_predictor_unavailable(loaded_extractor, monkeypatch):
    """Verify /ws/sign rejects connection with code 1013 when predictor is unavailable."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from app.main import app
    from app.models.predictor import predictor

    with TestClient(app) as client:
        monkeypatch.setattr(predictor, "_loaded", False)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/sign"):
                pass
        assert exc_info.value.code == 1013


def test_websocket_rejection_when_extractor_unavailable(loaded_predictor, monkeypatch):
    """Verify /ws/sign rejects connection with code 1013 when extractor is unavailable."""
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from app.main import app
    from app.recognition.landmark_extractor import extractor

    with TestClient(app) as client:
        monkeypatch.setattr(extractor, "_loaded", False)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/sign"):
                pass
        assert exc_info.value.code == 1013


@requires_model
def test_websocket_binary_and_base64_frame_input(loaded_predictor, loaded_extractor):
    """Verify binary JPEG/PNG frames and JSON Base64 frames (with and without prefix) are processed."""
    import cv2
    from fastapi.testclient import TestClient
    from app.main import app

    blank = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".jpg", blank)
    jpeg_bytes = buf.tobytes()
    b64_raw = base64.b64encode(jpeg_bytes).decode("ascii")
    b64_with_prefix = f"data:image/jpeg;base64,{b64_raw}"

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            init_msg = ws.receive_json()
            assert init_msg["type"] == "status"
            assert init_msg["status"] == "connected"

            # 1. Binary JPEG frame
            ws.send_bytes(jpeg_bytes)
            bin_res = ws.receive_json()
            assert bin_res["type"] == "status"
            assert bin_res["status"] == "collecting"
            assert bin_res["frames_collected"] == 1

            # 2. Base64 frame
            ws.send_text(json.dumps({"type": "frame", "data": b64_raw}))
            b64_res = ws.receive_json()
            assert b64_res["type"] == "status"
            assert b64_res["status"] == "collecting"
            assert b64_res["frames_collected"] == 2

            # 3. Base64 frame with data URI prefix
            ws.send_text(json.dumps({"type": "frame", "data": b64_with_prefix}))
            prefix_res = ws.receive_json()
            assert prefix_res["type"] == "status"
            assert prefix_res["status"] == "collecting"
            assert prefix_res["frames_collected"] == 3


@requires_model
def test_websocket_invalid_frames_and_disconnect_cleanup(loaded_predictor, loaded_extractor):
    """Verify invalid frames and messages return structured error without crashing or killing session."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            _ = ws.receive_json()  # Consume initial connected status

            # 1. Garbage binary bytes
            ws.send_bytes(b"not an image at all")
            err1 = ws.receive_json()
            assert err1["type"] == "error"
            assert err1["code"] == "INVALID_FRAME"

            # 2. Non-dict JSON text
            ws.send_text(json.dumps(["not", "an", "object"]))
            err2 = ws.receive_json()
            assert err2["type"] == "error"
            assert err2["code"] == "INVALID_FRAME"

            # 3. Missing frame data
            ws.send_text(json.dumps({"type": "frame", "data": ""}))
            err3 = ws.receive_json()
            assert err3["type"] == "error"
            assert err3["code"] == "INVALID_FRAME"

            # 4. Unknown control action
            ws.send_text(json.dumps({"type": "control", "action": "unknown_action"}))
            err4 = ws.receive_json()
            assert err4["type"] == "error"
            assert err4["code"] == "INVALID_FRAME"

            # 5. Unknown message type
            ws.send_text(json.dumps({"type": "unknown_type"}))
            err5 = ws.receive_json()
            assert err5["type"] == "error"
            assert err5["code"] == "INVALID_FRAME"

            # 6. Session still alive and accepts reset
            ws.send_text(json.dumps({"type": "control", "action": "reset"}))
            reset_res = ws.receive_json()
            assert reset_res["type"] == "status"
            assert reset_res["status"] == "reset"
            assert reset_res["frames_collected"] == 0


@requires_model
def test_websocket_full_32_frame_prediction_pipeline(loaded_predictor, loaded_extractor):
    """Verify feeding 32 frames over WebSocket triggers prediction matching Flutter client contract."""
    import cv2
    from fastapi.testclient import TestClient
    from app.main import app

    blank = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".jpg", blank)
    jpeg_bytes = buf.tobytes()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/sign") as ws:
            _ = ws.receive_json()  # Initial connected status

            # Send 31 frames -> all collecting
            for i in range(1, 32):
                ws.send_bytes(jpeg_bytes)
                msg = ws.receive_json()
                assert msg["type"] == "status"
                assert msg["status"] == "collecting"
                assert msg["frames_collected"] == i

            # 32nd frame -> triggers prediction
            ws.send_bytes(jpeg_bytes)
            pred = ws.receive_json()
            assert pred["type"] == "prediction"
            assert "prediction" in pred
            assert "confidence" in pred
            assert "recognized" in pred
            assert "top_predictions" in pred
            assert isinstance(pred["top_predictions"], list)
            assert len(pred["top_predictions"]) == 3
            assert "hands_detected" in pred
            assert pred["frames_collected"] == 32
            assert "timestamp" in pred
            assert "sentence" in pred
            assert "sentence_words" in pred
            assert "sentence_updated" in pred


