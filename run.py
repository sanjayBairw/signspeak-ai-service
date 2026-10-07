"""
SignSpeak AI Service — Entry point
===================================
Run with:
    python run.py

or directly with uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        # WebSocket ping/pong to detect dead connections
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )


if __name__ == "__main__":
    main()
