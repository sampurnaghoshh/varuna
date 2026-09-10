"""Track building and transmission-gap detection — §10 P0-5.

Gaps feed attribution channel E4, which treats a dark period as a boost only and
never penalises a clean transmitter (§5.3).
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession


async def build_tracks(
    session: AsyncSession,
    t_from: datetime,
    t_to: datetime,
    max_gap_min: float,
) -> int: ...


async def longest_gap_min(
    session: AsyncSession,
    mmsi: int,
    window_start: datetime,
    window_end: datetime,
) -> float: ...


async def interpolate_position(
    session: AsyncSession,
    mmsi: int,
    t: datetime,
) -> tuple[float, float] | None: ...
