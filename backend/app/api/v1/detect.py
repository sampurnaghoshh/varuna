"""Detection endpoints — §7.

    POST /detect/{scene_id}     detections + p_oil + SHAP factors
    GET  /detections/{id}
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.post("/detect/{scene_id}")
async def detect(scene_id: int) -> dict[str, Any]: ...


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: int) -> dict[str, Any]: ...
