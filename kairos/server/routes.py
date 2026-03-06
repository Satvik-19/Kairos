"""REST API endpoints for Kairos server mode."""
import base64
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()


def _get_engine():
    from .main import engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine


@router.get("/entropy")
def get_entropy():
    """Return a raw entropy sample from the current pool."""
    eng = _get_engine()
    raw = eng.pool.read(32)
    return {
        "entropy_hex": raw.hex(),
        "entropy_b64": base64.b64encode(raw).decode(),
        "pool_fill_percent": round(eng.pool.fill_percent(), 2),
        "sources": ["double_pendulum", "lorenz", "reaction_diffusion"],
    }


@router.get("/token")
def get_token(
    length: int = Query(default=32, ge=1, le=512),
    format: str = Query(default="hex"),
):
    """Return a secure random token. Returns 503 if pool health is critical."""
    eng = _get_engine()

    # Circuit breaker: suspend token generation when health is critical
    health = eng.health_monitor.get_cached()
    if health.get("health_status") == "critical":
        return JSONResponse(
            status_code=503,
            content={
                "error": "entropy_pool_degraded",
                "message": "Entropy pool health is critical. Token generation suspended.",
                "health": {
                    "entropy_score": health.get("entropy_score"),
                    "health_status": "critical",
                },
            },
        )

    fmt_map = {"hex": "hex", "base64": "base64", "uuid": "uuid"}
    fmt = fmt_map.get(format, "hex")
    token_value = eng.token(length, fmt)

    return {
        "token": token_value,
        "format": fmt,
        "length": length,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api-key")
def get_api_key():
    """Return a krs_-prefixed API key."""
    eng = _get_engine()
    return {
        "api_key": eng.api_key(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/nonce")
def get_nonce():
    """Return a single-use nonce value."""
    eng = _get_engine()
    nonce_hex = eng.nonce()
    nonce_int = int(nonce_hex, 16)
    return {
        "nonce": nonce_hex,
        "nonce_int": nonce_int,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health")
def get_health():
    """Return engine status and entropy health metrics."""
    eng = _get_engine()
    health = eng.health_monitor.get_cached()
    last_perturb = eng.perturber.last_perturbation_at()

    return {
        "status": "ok",
        "engines": {
            "double_pendulum":    "running" if eng.pendulum._running else "stopped",
            "lorenz":             "running" if eng.lorenz._running   else "stopped",
            "reaction_diffusion": "running" if eng.rd._running       else "stopped",
        },
        "pool_fill_percent":       round(eng.pool.fill_percent(), 2),
        "entropy_score":           health.get("entropy_score"),
        "distribution_uniformity": health.get("distribution_uniformity"),
        "duplicate_rate":          health.get("duplicate_rate"),
        "health_status":           health.get("health_status"),
        "last_perturbation_at":    last_perturb,
        "uptime_seconds":          eng.uptime_seconds(),
        # ML-enhanced fields (None / False until models are trained)
        "ml_active":               health.get("ml_active", False),
        "health_confidence":       health.get("health_confidence"),
        "anomaly_score":           health.get("anomaly_score"),
        "is_anomaly":              health.get("is_anomaly"),
        "prediction_resistance":   health.get("prediction_resistance"),
    }
