"""Scene request/response models. Mirrored by frontend/types/scene.ts (§8)."""

from pydantic import BaseModel

from app.schemas.common import Envelope


class SceneSummary(BaseModel): ...


class SceneDetail(SceneSummary): ...


class SceneListResponse(Envelope): ...


class SceneIngestResponse(Envelope): ...
