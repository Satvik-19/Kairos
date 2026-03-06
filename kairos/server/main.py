"""
FastAPI application factory for Kairos server mode.
Wraps the EntropyEngine library — the library has no FastAPI dependency.
"""
import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..engine import EntropyEngine
from .routes import router
from .websocket import ws_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance (shared across routes and websocket handlers)
engine: EntropyEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = EntropyEngine()
    health = engine.health()
    logger.info("KAIROS engine started | initial health: %s", health.get("health_status", "initializing"))
    yield
    if engine:
        engine.shutdown()
    logger.info("KAIROS engine stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kairos Entropy Engine",
        description="Chaos-driven entropy engine for secure token generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.include_router(ws_router)

    return app


app = create_app()


def run():
    """Entry point for `kairos-server` CLI command."""
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("kairos.server.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
