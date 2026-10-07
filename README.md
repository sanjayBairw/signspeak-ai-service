# SignSpeak AI Service

Production-ready Python AI service for real-time Indian Sign Language (ISL) recognition.

## Architecture

```
Flutter
  │
  ├── Existing Video Call ──────────► Agora  [UNCHANGED]
  │
  └── Sign Recognition ─────────────► Python AI Service (this)
                                           │
                                           ├── FastAPI + WebSocket
                                           ├── MediaPipe HandLandmarker
                                           ├── 32-frame sequence buffer
                                           └── 96-class Keras model
```

The AI recognition system is **completely independent** from Agora.

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.11 |
| TensorFlow | 2.17.1 |
| MediaPipe | 0.10.21 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |

## Model Files

The following files must exist (set via environment variables):

| File | Default Path |
|---|---|
| `signspeak_combined_best.keras` | `D:\Downloads\archive\processed\models_combined\` |
| `feature_mean.npy` | same directory |
| `feature_std.npy` | same directory |
| `class_names.json` | same directory |
| `hand_landmarker.task` | `D:\Downloads\archive\` |

## Setup

### 1. Create a Python 3.11 virtual environment

```bash
# Windows PowerShell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and update model paths if needed:

```bash
copy .env.example .env
```

The default `.env` already points to `D:\Downloads\archive\...`.

## Running

```bash
# From D:\signspeak\signspeak-main\ai_service\
python run.py
```

Or directly with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "signspeak-ai",
  "model": "signspeak_combined_best",
  "classes": 96
}
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/classes` | All 96 sign class names |
| `WebSocket` | `/ws/sign` | Real-time sign recognition |
| `GET` | `/docs` | Swagger UI |

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/sign`

### Client → Server

**Binary frame** (preferred):
```
Raw JPEG/PNG bytes
```

**JSON frame** (alternative):
```json
{ "type": "frame", "data": "<base64-encoded-image>" }
```

**Control messages:**
```json
{ "type": "control", "action": "clear_sentence" }
{ "type": "control", "action": "reset" }
```

### Server → Client

**While collecting frames (< 32):**
```json
{
  "type": "status",
  "status": "collecting",
  "frames_collected": 17,
  "required_frames": 32
}
```

**After 32 frames — prediction:**
```json
{
  "type": "prediction",
  "prediction": "Hello",
  "confidence": 97.42,
  "recognized": true,
  "top_predictions": [
    { "class": "Hello",     "confidence": 97.42 },
    { "class": "Thank you", "confidence": 1.21 },
    { "class": "Goodbye",   "confidence": 0.81 }
  ],
  "hands_detected": 2,
  "frames_collected": 32,
  "timestamp": 1234567890,
  "sentence": "Hello World",
  "sentence_words": ["Hello", "World"],
  "sentence_updated": true
}
```

**If confidence < threshold (60%):**
```json
{
  "prediction": "UNKNOWN",
  "recognized": false,
  "confidence": 42.3
}
```

## Model Specification

| Property | Value |
|---|---|
| Input shape | `(1, 32, 126)` |
| Sequence length | 32 frames |
| Features per frame | 126 (2 hands × 21 landmarks × XYZ) |
| Classes | 96 |
| Architecture | Conv1D → BiGRU → BiGRU → Attention → Dense → Softmax |
| Normalization | `(x - mean) / std` per-feature z-score |

## Preprocessing (matches training exactly)

1. Receive frame (JPEG/PNG)
2. Resize preserving aspect ratio (width → 384px)
3. BGR → RGB
4. MediaPipe HandLandmarker (IMAGE mode, `detect()`)
5. Extract up to 2 hands
6. Wrist-relative coordinates: `lm.xyz - wrist.xyz`
7. Sort hands left-to-right by wrist X
8. Slot 0 = leftmost, slot 1 = rightmost, missing = zeros
9. Flatten → 126 values
10. Add to rolling deque(maxlen=32)
11. Normalize with `feature_mean.npy` / `feature_std.npy`
12. Run model with `training=False`

## Configuration

All settings in `.env`:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Service port |
| `CONFIDENCE_THRESHOLD` | `60` | Min % for recognized prediction |
| `PREDICTION_INTERVAL` | `0.08` | Seconds between predictions |
| `DUPLICATE_COOLDOWN` | `2.0` | Seconds before same word repeats |
| `INFERENCE_WIDTH` | `384` | Frame resize width |
| `NUM_HANDS` | `2` | Max hands to detect |

## Testing

```bash
cd D:\signspeak\signspeak-main\ai_service
pytest tests/ -v
```

Tests cover:
1. Model loading
2. 96-class verification
3. Landmark extraction shape `(126,)`
4. Sequence shape `(32, 126)`
5. Normalization
6. Full prediction pipeline
7. Top-3 predictions
8. Unknown threshold
9. WebSocket connection
10. Invalid frame handling
11. Hello video sanity test

## Node.js Integration

The existing Node.js backend exposes these endpoints:

| Endpoint | Description |
|---|---|
| `GET /api/v1/ai/health` | AI service health |
| `GET /api/v1/ai/classes` | Sign class list |
| `GET /api/v1/ai/ws-url` | WebSocket URL for Flutter |

These are served by:
- `mutedbackend/src/services/aiService.js`
- `mutedbackend/src/controllers/aiController.js`
- `mutedbackend/src/routes/aiRoutes.js`

**If the AI service is offline**, Node.js returns graceful degraded responses.
**All existing Node.js functionality (auth, chat, calls, Agora) continues working.**

## Flutter Integration

New Flutter feature at:
```
lib/features/ai_sign_recognition/
├── models/sign_prediction_model.dart
├── providers/sign_recognition_provider.dart
├── screens/sign_recognition_screen.dart
├── services/ai_sign_recognition_service.dart
└── widgets/prediction_display_widget.dart
```

Navigate to the screen:
```dart
Navigator.pushNamed(context, SignRecognitionScreen.routeName);
// or with go_router:
context.push('/sign-recognition');
```

## Agora Status

> **Agora was not modified.**
>
> Zero Agora files were changed. Zero Agora packages changed.
> The AI recognition system is a fully independent feature.
