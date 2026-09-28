"""FastAPI — REST (sesje, modele, health) + WebSocket na żywo.

Uruchomienie dev:
  uvicorn cr_rl.server.app:app --reload --port 8000

Produkcja: serwuje zbudowany frontend z web/dist (SPA fallback).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cr_rl.coach.inference import DEFAULT_MODEL_PATH, PolicyInspector
from cr_rl.paths import MODELS_DIR, WEB_DIST_DIR
from cr_rl.server.protocol import (
    ErrorMsg,
    PauseMsg,
    PingMsg,
    PlayCardMsg,
    PongMsg,
    RematchMsg,
    SpeedMsg,
    parse_client_message,
)
from cr_rl.server.sessions import VALID_MODES, GameSession, SessionManager
from cr_rl.server.training_metrics import cached_training_metrics

logger = logging.getLogger(__name__)

WEB_DIST = WEB_DIST_DIR
DEPLOY_MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))


class CreateSessionRequest(BaseModel):
    mode: str = Field(default="coach")
    lang: str = Field(default="pl", pattern="^(pl|en)$")
    seed: Optional[int] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    mode: str
    coach_enabled: bool


def create_app(
    inspector: Optional[PolicyInspector] = None,
    *,
    model_path: str | Path | None = DEPLOY_MODEL_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inspector = inspector or PolicyInspector(model_path)
        app.state.sessions = SessionManager(app.state.inspector)
        logger.info(
            "Serwer wystartował; model RL: %s",
            app.state.inspector.model_path or "brak (trener heurystyczny)",
        )
        yield
        for session_id in list(app.state.sessions._sessions):
            app.state.sessions.discard(session_id)

    app = FastAPI(title="Clash Royale RL Coach", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- REST ---

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": app.state.inspector.available,
            "model_path": str(app.state.inspector.model_path) if app.state.inspector.model_path else None,
            "active_sessions": len(app.state.sessions),
        }

    @app.get("/api/models")
    def models() -> dict:
        files = sorted(MODELS_DIR.glob("**/*.zip"))
        return {"models": [str(f) for f in files]}

    @app.get("/api/training/metrics")
    def training_metrics() -> dict:
        return cached_training_metrics()

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
        if request.mode not in VALID_MODES:
            raise HTTPException(status_code=422, detail=f"Nieznany tryb: {request.mode!r}")
        session = app.state.sessions.create(request.mode, lang=request.lang, seed=request.seed)
        return CreateSessionResponse(
            session_id=session.id,
            mode=session.mode,
            coach_enabled=session.coach is not None,
        )

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict:
        if app.state.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="Sesja nie istnieje")
        app.state.sessions.discard(session_id)
        return {"status": "closed"}

    # --- WebSocket ---

    @app.websocket("/ws/sessions/{session_id}")
    async def session_ws(websocket: WebSocket, session_id: str) -> None:
        session: Optional[GameSession] = app.state.sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="Sesja nie istnieje")
            return

        await websocket.accept()
        outbox: asyncio.Queue = asyncio.Queue(maxsize=1000)
        outbox.put_nowait(session.hello().model_dump(exclude_none=True))
        outbox.put_nowait(session.build_snapshot().model_dump(exclude_none=True))

        game_task = asyncio.create_task(session.run(outbox))
        sender = asyncio.create_task(_sender_loop(websocket, outbox))
        try:
            while True:
                data = await websocket.receive_json()
                response = await _handle_client_message(session, data)
                for message in response:
                    outbox.put_nowait(message.model_dump(exclude_none=True))
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Błąd połączenia WS (sesja %s)", session_id)
        finally:
            session.stop()
            game_task.cancel()
            sender.cancel()
            app.state.sessions.discard(session_id)

    async def _sender_loop(websocket: WebSocket, outbox: asyncio.Queue) -> None:
        try:
            while True:
                message = await outbox.get()
                await websocket.send_json(message)
        except (asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
            pass

    async def _handle_client_message(session: GameSession, data: dict) -> list:
        try:
            message = parse_client_message(data)
        except ValueError as error:
            return [ErrorMsg(message=str(error))]

        loop = asyncio.get_running_loop()
        if isinstance(message, PlayCardMsg):
            # ocena trenera + inferencja — poza pętlą zdarzeń
            return await loop.run_in_executor(
                None, session.handle_play, message.card, message.x, message.y
            )
        if isinstance(message, PauseMsg):
            session.set_paused(message.paused)
        elif isinstance(message, SpeedMsg):
            session.set_speed(message.value)
        elif isinstance(message, RematchMsg):
            await loop.run_in_executor(None, session.rematch)
        elif isinstance(message, PingMsg):
            return [PongMsg()]
        return []

    # --- statyczny frontend (produkcja) ---

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            if full_path.startswith(("api/", "ws/")):
                raise HTTPException(status_code=404)
            candidate = WEB_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("cr_rl.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
