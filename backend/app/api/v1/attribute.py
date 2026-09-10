"""Attribution endpoints — §7.

    POST /attribute/{detection_id}      ranked candidates + channel breakdown + verdict
    GET  /vessels/{mmsi}/track?from&to

A sub-threshold candidate is never returned as a culprit (§5.4). UNATTRIBUTED
carries a queued case and a cross-check recommendation, and is a 200, not an
error.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.post("/attribute/{detection_id}")
async def attribute(detection_id: int) -> dict[str, Any]: ...


@router.get("/vessels/{mmsi}/track")
async def get_vessel_track(
    mmsi: int,
    from_ts: datetime,
    to_ts: datetime,
) -> dict[str, Any]: ...
