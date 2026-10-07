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
# Dataset / model paths (same as .env defaults)
# ─────────────────────────────────────────────────────────────

BASE = r"D:\Downloads\archive"
MODEL_PATH = os.path.join(BASE, "processed", "models_combined", "signspeak_combined_best.keras")
MEAN_PATH  = os.path.join(BASE, "processed", "models_combined", "feature_mean.npy")
STD_PATH   = os.path.join(BASE, "processed", "models_combined", "feature_std.npy")
CLASS_PATH = os.path.join(BASE, "processed", "models_combined", "class_names.json")
HAND_MODEL_PATH = os.path.join(BASE, "hand_landmarker.task")

# Known "Hello" video for the sanity test (pick first available)
HELLO_VIDEO_CANDIDATES = [
    os.path.join(BASE, "Video_Dataset", "Video_Dataset", "Hello", "Hello_1.mp4"),
    os.path.join(BASE, "Video_Dataset", "Video_Dataset", "Hello", "Hello_2.mp4"),
    os.path.join(BASE, "Sample Videos", "Hello.mp4"),
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

    from app.models.predictor import SignPredictor
    p = SignPredictor()
    p.load()
    return p


@pytest.fixture(scope="session")
def loaded_extractor():
    """Load the MediaPipe extractor once for the whole test session."""
    from app.recognition.landmark_extractor import LandmarkExtractor
    e = LandmarkExtractor()
    e.load()
    return e


@pytest.fixture(scope="session")
def class_names(loaded_predictor):
    return loaded_predictor.class_names


# ─────────────────────────────────────────────────────────────
# 1. Model loading
# ─────────────────────────────────────────────────────────────

@requires_model
def test_model_loading(loaded_predictor):
    assert loaded_predictor.is_loaded, "Predictor should be loaded"


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
    import numpy as np
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    features, hands_detected = loaded_extractor.extract_from_bgr(blank)
    assert features.shape == (126,), f"Feature shape wrong: {features.shape}"
    assert hands_detected == 0


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


# ─────────────────────────────────────────────────────────────
# 11. Known video sanity test — Hello should be in top predictions
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
        try:
            feats, _ = loaded_extractor.extract_from_bgr(frame)
            feature_list.append(feats)
        except Exception:
            feature_list.append(np.zeros(126, dtype=np.float32))

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
