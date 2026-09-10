"""Response envelope shared by every §7 endpoint.

Every response carries {"source": "live" | "fixture", "elapsed_ms": int}. The
frontend renders the source badge from it — a fixture-backed answer is never
presented as a live one (§2.2).
"""

from typing import Literal

from pydantic import BaseModel

Source = Literal["live", "fixture"]


class Envelope(BaseModel):
    source: Source
    elapsed_ms: int


class ErrorResponse(Envelope):
    """Typed degradation, not an exception surfaced to the user (§8)."""

    stage: str
    detail: str
