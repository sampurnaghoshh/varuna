"""Drift endpoints — §7.

    POST /drift/backward            {detection_id, horizon_h, windage, k_h, theta_dev}
    POST /drift/forward             {detection_id, horizon_h}
    GET  /drift/{run_id}/frames     scrubber frames, step_min default 30
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.post("/drift/backward")
async def drift_backward(payload: dict[str, Any]) -> dict[str, Any]: ...


@router.post("/drift/forward")
async def drift_forward(payload: dict[str, Any]) -> dict[str, Any]: ...


@router.get("/drift/{run_id}/frames")
async def get_frames(run_id: int, step_min: int = 30) -> dict[str, Any]: ...
