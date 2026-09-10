"""Scene endpoints — §7.

    GET  /scenes            list bundled scenes
    POST /scenes/ingest     upload a GeoTIFF
    GET  /scenes/{id}

Scene ingest is deliberately split-unaware (§0): the demo pipeline is supposed to
load holdout scenes, and a guard here would either break the demo or teach
everyone to bypass it. The split is enforced at the training-data loader only.
"""

from typing import Any

from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.get("/scenes")
async def list_scenes() -> dict[str, Any]: ...


@router.post("/scenes/ingest")
async def ingest_scene(file: UploadFile) -> dict[str, Any]: ...


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: int) -> dict[str, Any]: ...
