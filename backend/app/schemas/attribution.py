"""Attribution request/response models. Mirrored by frontend/types/attribution.ts.

An UNATTRIBUTED response is a normal 200 carrying a queued case and a
cross-check recommendation. It never names a top-ranked vessel as a culprit
(§5.4).
"""

from pydantic import BaseModel

from app.schemas.common import Envelope


class ChannelBreakdown(BaseModel): ...


class CandidateOut(BaseModel): ...


class AttributionResponse(Envelope): ...


class VesselTrackResponse(Envelope): ...
