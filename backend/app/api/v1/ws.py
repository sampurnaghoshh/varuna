"""Pipeline WebSocket — §7.

    WS /ws/pipeline/{job_id}

Event shape the UI depends on exactly:

    {"stage": "SEGMENTING" | "DISCRIMINATING" | "REWINDING" | "FUSING" | "DONE"
              | "ERROR",
     "progress": 0.0-1.0, "message": "human readable", "payload": {}}

Stage messages are what make the judge watch the system think. Write them well:
"Rewinding ocean state to T-4h 30m", not "Processing...".

Events are fanned out from Redis pub/sub (§3).
"""

from typing import Any, Literal

from fastapi import APIRouter, WebSocket

router = APIRouter()

Stage = Literal["SEGMENTING", "DISCRIMINATING", "REWINDING", "FUSING", "DONE", "ERROR"]


@router.websocket("/ws/pipeline/{job_id}")
async def pipeline_socket(websocket: WebSocket, job_id: str) -> None: ...


async def publish_stage(
    job_id: str,
    stage: Stage,
    progress: float,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None: ...
