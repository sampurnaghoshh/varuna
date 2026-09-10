"""Detection request/response models. Mirrored by frontend/types/detection.ts.

Any accuracy figure carried to the UI travels with its split name and n (§16).
"""

from pydantic import BaseModel

from app.schemas.common import Envelope


class DetectionFeaturesOut(BaseModel): ...


class ShapFactor(BaseModel): ...


class DetectionOut(BaseModel): ...


class DetectResponse(Envelope): ...
