"""Lagrangian RK4 drift solver, forward and backward — §5.1.

    v(x, t) = u_current(x, t) + alpha * u_wind10(x, t) + u_stokes(x, t)

Backward mode integrates -v, forward mode +v. Turbulent diffusion is added each
step in BOTH modes: it is symmetric, and dropping it on the reverse leg would
understate the origin uncertainty, which is the one number this project exists
to report honestly.

    dx_diff ~ N(0, sigma),  sigma = sqrt(2 * K_h * dt)

Defaults (config.py): alpha 0.033, theta_dev 0 deg, dt 300 s, horizon 12 h,
5000 particles, K_h 10 m^2/s, snapshot every 30 min.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely.geometry import Polygon

Mode = Literal["forward", "backward"]


@dataclass(frozen=True)
class DriftState:
    t_offset_min: int
    positions: np.ndarray
    stretch_factor: float


@dataclass(frozen=True)
class DriftRun:
    mode: Mode
    n_particles: int
    horizon_h: int
    windage: float
    k_h: float
    theta_dev_deg: float
    field_source: str
    states: list[DriftState]


def seed_particles(polygon: Polygon, n_particles: int, seed: int) -> np.ndarray:
    """Area-weighted uniform sampling inside the detected slick polygon."""
    ...


def velocity(
    positions: np.ndarray,
    t_s: float,
    fields: object,
    windage: float,
    theta_dev_deg: float,
) -> np.ndarray: ...


def rk4_step(
    positions: np.ndarray,
    t_s: float,
    dt_s: float,
    fields: object,
    sign: int,
) -> np.ndarray: ...


def diffusion_step(
    positions: np.ndarray,
    k_h: float,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray: ...


def stretch_factor(positions: np.ndarray) -> float:
    """Ratio of the particle cloud's major axis now to its major axis at seed
    time. Attribution channel E3 divides the observed slick length by this to
    recover the released length."""
    ...


def run(
    polygon: Polygon,
    mode: Mode,
    fields: object,
    horizon_h: int,
    n_particles: int,
    windage: float,
    k_h: float,
    theta_dev_deg: float,
    dt_s: int,
    snapshot_interval_min: int,
    seed: int,
) -> DriftRun: ...
