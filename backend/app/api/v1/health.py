"""GET /health — liveness plus a real dependency probe.

Returns 200 whether or not the dependencies answer (§8: nothing raises to the
user). `status` is "ok" only when every dependency responded; the container
healthcheck asserts that field, so a half-up stack does not report healthy.
"""

import time
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.db.session import SessionLocal

router = APIRouter()


async def _check_postgis() -> dict[str, Any]:
    try:
        async with SessionLocal() as session:
            version = await session.scalar(text("SELECT PostGIS_Lib_Version()"))
        return {"up": True, "postgis_version": version}
    except Exception as exc:  # noqa: BLE001 - degrade, never raise (§8)
        return {"up": False, "error": type(exc).__name__}


async def _check_redis() -> dict[str, Any]:
    client = aioredis.from_url(settings.redis_url)
    try:
        await client.ping()
        return {"up": True}
    except Exception as exc:  # noqa: BLE001 - degrade, never raise (§8)
        return {"up": False, "error": type(exc).__name__}
    finally:
        await client.aclose()


@router.get("/health")
async def health() -> dict[str, Any]:
    started = time.perf_counter()
    db = await _check_postgis()
    cache = await _check_redis()
    healthy = db["up"] and cache["up"]
    return {
        "status": "ok" if healthy else "degraded",
        "app": settings.app_name,
        "env": settings.env,
        "services": {"postgis": db, "redis": cache},
        "source": "live",
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
