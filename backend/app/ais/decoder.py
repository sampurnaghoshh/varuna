"""AIS decoding — AIVDM sentences and the Danish Maritime Authority CSV export.

The Baltic demo (§12 SC-01) runs on the Danish CSV; the AIVDM path exists for
any live or replayed NMEA stream.
"""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class AisPosition(TypedDict):
    mmsi: int
    ts: datetime
    lon: float
    lat: float
    sog: float | None
    cog: float | None
    heading: float | None
    nav_status: str | None


class AisStatic(TypedDict):
    mmsi: int
    imo: int | None
    name: str | None
    callsign: str | None
    type: str | None
    length_m: float | None
    width_m: float | None
    flag: str | None


def decode_aivdm(sentence: str) -> AisPosition | AisStatic | None: ...


def read_danish_csv(path: Path) -> Iterator[tuple[AisStatic, AisPosition]]: ...
