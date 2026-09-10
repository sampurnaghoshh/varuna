"""Wind and current field loaders with a synthetic fallback — §5.1.

Resolution order, logged and reported as `field_source` in every drift response:

    1. cached CMEMS/ERA5 NetCDF under data/
    2. bundled per-scenario field snapshot
    3. synthetic field

A missing field never crashes a run. It degrades and says which source it used.

The synthetic field is geostrophic-like flow with 2-3 mesoscale eddies plus a
spatially coherent wind, and is deterministic given a seed — the demo must
reproduce identically offline (§2.1).
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

FieldSource = Literal["cmems_netcdf", "scenario_snapshot", "synthetic"]


class VelocityField(Protocol):
    source: FieldSource

    def current_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...

    def wind10_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...

    def stokes_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...


def load_netcdf(path: Path) -> VelocityField | None: ...


def load_scenario_snapshot(scenario_code: str) -> VelocityField | None: ...


def synthetic_field(bbox: tuple[float, float, float, float], seed: int) -> VelocityField: ...


def resolve(
    bbox: tuple[float, float, float, float],
    t0: datetime,
    scenario_code: str | None,
    seed: int,
) -> VelocityField: ...
