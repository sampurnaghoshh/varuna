"""Demo-mode endpoints — §7, §12.

    POST /demo/run/{scenario_code}      fires the WS pipeline event sequence
    POST /demo/reset

Scenario codes: SC-01 (Baltic Night Discharge), SC-02 (The Look-alike Trap,
P0-CRITICAL), SC-03 (Clean Sea). Total runtime <= 170 s.
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.post("/demo/run/{scenario_code}")
async def run_scenario(scenario_code: str) -> dict[str, Any]: ...


@router.post("/demo/reset")
async def reset_demo() -> dict[str, Any]: ...
