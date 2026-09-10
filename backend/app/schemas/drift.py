"""Drift request/response models. Mirrored by frontend/types/drift.ts.

Tunable bounds mirror config.py: windage [0.020, 0.040], theta_dev [-20, 20].
`field_source` is always reported so the UI can say which ocean state was used.
"""

from pydantic import BaseModel

from app.schemas.common import Envelope


class DriftBackwardRequest(BaseModel): ...


class DriftForwardRequest(BaseModel): ...


class DriftFrame(BaseModel): ...


class DriftRunResponse(Envelope): ...


class DriftFramesResponse(Envelope): ...
