"""WebSocket endpoints for live chaos state and entropy streams."""
import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()


def _get_engine():
    from .main import engine
    return engine


@ws_router.websocket("/ws/chaos")
async def ws_chaos(websocket: WebSocket):
    """
    Stream live chaos system state to the dashboard at ~20fps (50ms intervals).
    Sends pendulum, lorenz, and reaction_diffusion states.
    """
    await websocket.accept()
    eng = _get_engine()
    if eng is None:
        await websocket.close(code=1011)
        return

    try:
        while True:
            states = eng.get_engine_states()
            payload = {
                "t": time.time(),
                "pendulum": states["pendulum"],
                "lorenz": states["lorenz"],
                "reaction_diffusion": states["reaction_diffusion"],
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.05)  # ~20fps
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@ws_router.websocket("/ws/entropy")
async def ws_entropy(websocket: WebSocket):
    """
    Stream entropy pool stats and health metrics at ~2fps (500ms intervals).
    """
    await websocket.accept()
    eng = _get_engine()
    if eng is None:
        await websocket.close(code=1011)
        return

    try:
        while True:
            health = eng.health_monitor.get_cached()
            payload = {
                "t":                       time.time(),
                "pool_fill_percent":       round(eng.pool.fill_percent(), 2),
                "entropy_score":           health.get("entropy_score", 0.0),
                "distribution_uniformity": health.get("distribution_uniformity", 0.0),
                "duplicate_rate":          health.get("duplicate_rate", 0.0),
                "health_status":           health.get("health_status", "degraded"),
                "tokens_generated_total":  eng.tokens_generated(),
                "hashes_per_second":       eng.hashes_per_second(),
                "entropy_rate_bps":        eng.entropy_rate_bps(),
                # ML-enhanced fields (None / False until models are trained)
                "ml_active":               health.get("ml_active", False),
                "health_confidence":       health.get("health_confidence"),
                "anomaly_score":           health.get("anomaly_score"),
                "is_anomaly":              health.get("is_anomaly"),
                "prediction_resistance":   health.get("prediction_resistance"),
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)  # ~2fps
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
