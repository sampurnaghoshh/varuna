"""AIS ingest into PostGIS — §10 P0-5.

Injected tracks are written with is_injected = true on both `vessels` and
`ais_positions`. The UI reads that flag to render the INJECTED - SIMULATED
badge (§2.3); nothing may write an injected row without it.
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession


async def ingest_danish_csv(
    session: AsyncSession,
    path: Path,
    bbox: tuple[float, float, float, float],
    t_from: datetime,
    t_to: datetime,
) -> int:
    """Returns the number of positions inserted."""
    ...


async def inject_track(
    session: AsyncSession,
    mmsi: int,
    positions: list[tuple[datetime, float, float, float, float]],
) -> None:
    """Writes a simulated track. is_injected is forced true (§2.3)."""
    ...
