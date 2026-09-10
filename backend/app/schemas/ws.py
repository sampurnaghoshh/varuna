"""WebSocket pipeline event model — §7.

The UI depends on this shape exactly. Changing a field name here breaks the
console mid-demo.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

Stage = Literal["SEGMENTING", "DISCRIMINATING", "REWINDING", "FUSING", "DONE", "ERROR"]


class PipelineEvent(BaseModel):
    stage: Stage
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
