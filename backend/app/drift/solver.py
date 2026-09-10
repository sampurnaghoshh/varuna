"""Lagrangian RK4 drift solver, forward and backward — §5.1.

    v(x, t) = u_current(x, t) + alpha * u_wind10(x, t) + u_stokes(x, t)

Backward mode integrates -v, forward mode +v. Turbulent diffusion is added each
step in BOTH modes: it is symmetric, and dropping it on the reverse leg would
understate the origin uncertainty, which is the one number this project exists
to report honestly.

    dx_diff ~ N(0, sigma),  sigma = sqrt(2 * K_h * dt)

Defaults (config.py): alpha 0.033, theta_dev 0 deg, dt 300 s, horizon 12 h,
5000 particles, K_h 10 m^2/s, snapshot every 30 min.

Frames: particle state is carried in metres in a local UTM zone chosen from the
slick centroid, never in degrees (§8). The field Protocol is queried in lon/lat,
so each RK4 substage transforms back. Grid convergence across a cloud of this
size is under a degree, so field east/north components are used directly as UTM
easting/northing rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from app.config import settings
from app.drift.fields import VelocityField

logger = logging.getLogger(__name__)

Mode = Literal["forward", "backward"]

_MAX_SEED_ROUNDS = 64


class DegenerateSlickError(ValueError):
    """Raised for a slick polygon with no usable area to seed particles in."""


@dataclass(frozen=True)
class Frame:
    """Local UTM working frame for one run, built once and reused.

    Distances, diffusion and PCA all happen here. Only the boundary converts.
    """

    crs: CRS
    to_utm: Transformer
    to_wgs84: Transformer

    @classmethod
    def for_point(cls, lon: float, lat: float) -> Frame:
        zone = int((lon + 180.0) // 6.0) % 60 + 1
        epsg = (32600 if lat >= 0.0 else 32700) + zone
        crs = CRS.from_epsg(epsg)
        wgs84 = CRS.from_epsg(4326)
        return cls(
            crs=crs,
            to_utm=Transformer.from_crs(wgs84, crs, always_xy=True),
            to_wgs84=Transformer.from_crs(crs, wgs84, always_xy=True),
        )

    def forward(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = self.to_utm.transform(lon, lat)
        return np.stack((np.asarray(x), np.asarray(y)), axis=-1)

    def inverse(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self.to_wgs84.transform(positions[:, 0], positions[:, 1])
        return np.asarray(lon), np.asarray(lat)


@dataclass(frozen=True)
class DriftState:
    t_offset_min: int
    positions: np.ndarray
    stretch_factor: float
    stretch_factor_raw: float


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
    t0: datetime
    frame: Frame
    seed: int
    elapsed_s: float

    def positions_wgs84(self, index: int) -> np.ndarray:
        lon, lat = self.frame.inverse(self.states[index].positions)
        return np.stack((lon, lat), axis=-1)


def seed_particles(polygon: Polygon, n_particles: int, seed: int) -> np.ndarray:
    """Area-weighted uniform sampling inside the detected slick polygon.

    Rejection sampling in the bounding box, which is area-weighted uniform by
    construction and holes-and-concavity correct. Triangulating instead would be
    faster, but shapely triangulates the convex hull, silently seeding particles
    outside a concave slick - and E1 integrates over exactly those particles.

    Returns lon/lat, EPSG:4326. Deterministic given `seed`.
    """
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}")
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
        raise DegenerateSlickError("Slick polygon is empty, invalid or has zero area")

    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    rng = np.random.default_rng(seed)
    shapely.prepare(polygon)

    kept_lon: list[np.ndarray] = []
    kept_lat: list[np.ndarray] = []
    n_kept = 0
    fill = max(polygon.area / max((lon_max - lon_min) * (lat_max - lat_min), 1e-30), 1e-3)

    rounds = 0
    while n_kept < n_particles:
        rounds += 1
        if rounds > _MAX_SEED_ROUNDS:
            raise DegenerateSlickError(
                f"Could not seed {n_particles} particles in {_MAX_SEED_ROUNDS} rounds; "
                "polygon area is vanishing relative to its bounds"
            )
        batch = int(np.ceil((n_particles - n_kept) / fill * 1.3)) + 32
        lon = rng.uniform(lon_min, lon_max, size=batch)
        lat = rng.uniform(lat_min, lat_max, size=batch)
        inside = shapely.contains_xy(polygon, lon, lat)
        if inside.any():
            kept_lon.append(lon[inside])
            kept_lat.append(lat[inside])
            n_kept += int(inside.sum())

    lon_all = np.concatenate(kept_lon)[:n_particles]
    lat_all = np.concatenate(kept_lat)[:n_particles]
    return np.stack((lon_all, lat_all), axis=-1)


def velocity(
    positions: np.ndarray,
    t_s: float,
    fields: VelocityField,
    windage: float,
    theta_dev_deg: float,
    frame: Frame,
) -> np.ndarray:
    """v = u_current + alpha * R(theta_dev) u_wind10 + u_stokes, in m/s.

    theta_dev is a Coriolis-driven veer of the wind-driven component relative to
    the wind vector: positive rotates clockwise, the Northern-Hemisphere sense.
    It applies to the windage term only - the current and Stokes terms already
    carry their own direction.
    """
    lon, lat = frame.inverse(positions)
    t = datetime.fromtimestamp(t_s, tz=UTC)

    current = fields.current_ms(lon, lat, t)
    wind = fields.wind10_ms(lon, lat, t)
    stokes = fields.stokes_ms(lon, lat, t)

    theta = np.radians(-theta_dev_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    wind_rot = np.stack(
        (
            wind[:, 0] * cos_t - wind[:, 1] * sin_t,
            wind[:, 0] * sin_t + wind[:, 1] * cos_t,
        ),
        axis=-1,
    )
    return current + windage * wind_rot + stokes


def rk4_step(
    positions: np.ndarray,
    t_s: float,
    dt_s: float,
    fields: VelocityField,
    sign: int,
    windage: float,
    theta_dev_deg: float,
    frame: Frame,
) -> np.ndarray:
    """One RK4 step. `sign` applies to both the velocity and the time advance.

    A backward step therefore evaluates the ocean at the actual, decreasing model
    time rather than reusing the forward-time field with a flipped sign. That is
    what makes a backward-then-forward round trip close, and it is what §5.1
    means by "integrates -v".
    """
    h = sign * dt_s

    k1 = velocity(positions, t_s, fields, windage, theta_dev_deg, frame)
    k2 = velocity(positions + 0.5 * h * k1, t_s + 0.5 * h, fields, windage, theta_dev_deg, frame)
    k3 = velocity(positions + 0.5 * h * k2, t_s + 0.5 * h, fields, windage, theta_dev_deg, frame)
    k4 = velocity(positions + h * k3, t_s + h, fields, windage, theta_dev_deg, frame)

    return positions + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def diffusion_step(
    positions: np.ndarray,
    k_h: float,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random-walk turbulent diffusion, sigma = sqrt(2 * K_h * dt), in metres.

    Applied in both modes. Diffusion is symmetric under time reversal: a reverse
    run that omitted it would return a false-precision origin, which is the exact
    failure §2.2 exists to prevent.
    """
    if k_h <= 0.0:
        return positions
    sigma = np.sqrt(2.0 * k_h * dt_s)
    return positions + rng.normal(0.0, sigma, size=positions.shape)


def major_axis_m(positions: np.ndarray, diffusive_variance_m2: float = 0.0) -> float:
    """PCA major-axis extent of the particle cloud, in metres.

    Two standard deviations along the leading eigenvector. The constant cancels
    in every ratio it is used for; what matters is that it is the same constant
    at both ends.

    `diffusive_variance_m2` is subtracted from the leading eigenvalue before the
    root. The random walk adds 2*K_h*t of variance isotropically - to both
    eigenvalues equally - independent of the advection, so removing it leaves the
    advective strain alone. Without this the "stretch" of a reverse run is mostly
    a measure of its own diffusion, which is uncertainty, not strain.
    """
    if positions.shape[0] < 2:
        return 0.0
    centred = positions - positions.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    advective = max(float(eigenvalues[-1]) - diffusive_variance_m2, 0.0)
    return float(2.0 * np.sqrt(advective))


def stretch_factor(
    positions: np.ndarray,
    seed_positions: np.ndarray,
    mode: Mode,
    k_h: float = 0.0,
    elapsed_s: float = 0.0,
) -> tuple[float, float]:
    """Cloud major axis, later in physical time / earlier in physical time.

    Backward: the seed IS the observed slick and the snapshot is the earlier,
    released patch, so the ratio is seed / snapshot. Forward: the seed is the
    release, so it is snapshot / seed. Either way the value is >= 1 for a
    spreading cloud, which is what makes E3's

        L_released = major_axis_km / stretch_factor

    shorten the observed slick back to the released patch (§5.3). Defining it as
    "now / seed" in both modes inverts the backward case and hands E3 a released
    length longer than the observation.

    Returns (reported, raw). `raw` is the bare ratio of diffusion-corrected axes.
    `reported` is that value floored at 1.0, because oil spreads and does not
    contract: a released patch longer than the observed slick is physically
    impossible, and E3 divides by this. When the flow's strain axes are not
    aligned with the observed slick the reverse run can widen the cloud along the
    other axis and drive the raw ratio below 1 - the floor refuses to turn that
    into a longer release, degrading E3 to "no stretch information" instead.

    Both values are carried on the snapshot. The floor is a stated physical
    bound, not a silent correction (§2.2): nothing is hidden from the UI.
    """
    diffusive = 2.0 * k_h * abs(elapsed_s)
    now = major_axis_m(positions, diffusive_variance_m2=diffusive)
    at_seed = major_axis_m(seed_positions)
    if at_seed <= 0.0 or now <= 0.0:
        return 1.0, 1.0
    raw = at_seed / now if mode == "backward" else now / at_seed
    return max(raw, 1.0), raw


def run(
    polygon: Polygon,
    mode: Mode,
    fields: VelocityField,
    t0: datetime | None = None,
    horizon_h: int | None = None,
    n_particles: int | None = None,
    windage: float | None = None,
    k_h: float | None = None,
    theta_dev_deg: float | None = None,
    dt_s: int | None = None,
    snapshot_interval_min: int | None = None,
    seed: int | None = None,
) -> DriftRun:
    """Integrate the cloud and emit a snapshot every `snapshot_interval_min`.

    Every parameter defaults to config.py but is overridable per run: the what-if
    sliders and a responsiveness-constrained demo need a reduced particle count
    without editing settings.

    `t_offset_min` is signed - a backward run emits 0, -30, ... , -720 - so the
    absolute snapshot time is t0 + t_offset_min. E1 records t*_j from it and the
    §7 stage messages render "T-4h 30m" straight off it.
    """
    started = datetime.now(tz=UTC)

    horizon_h = settings.drift_horizon_h if horizon_h is None else horizon_h
    n_particles = settings.drift_n_particles if n_particles is None else n_particles
    windage = settings.windage_alpha if windage is None else windage
    k_h = settings.k_h_m2s if k_h is None else k_h
    theta_dev_deg = settings.theta_dev_deg if theta_dev_deg is None else theta_dev_deg
    dt_s = settings.drift_dt_s if dt_s is None else dt_s
    snapshot_interval_min = (
        settings.drift_snapshot_interval_min
        if snapshot_interval_min is None
        else snapshot_interval_min
    )
    seed = settings.drift_seed if seed is None else seed
    t0 = datetime.now(tz=UTC) if t0 is None else t0

    if horizon_h <= 0:
        raise ValueError(f"horizon_h must be positive, got {horizon_h}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if not 0.020 <= windage <= 0.040:
        raise ValueError(f"windage {windage} outside the §5.1 range [0.020, 0.040]")
    if not -20.0 <= theta_dev_deg <= 20.0:
        raise ValueError(f"theta_dev_deg {theta_dev_deg} outside the §5.1 range [-20, +20]")

    snapshot_every_s = snapshot_interval_min * 60
    if snapshot_every_s % dt_s != 0:
        raise ValueError(
            f"snapshot interval {snapshot_interval_min} min is not a whole number of "
            f"{dt_s} s steps; snapshots would drift off the grid"
        )

    # Independent streams so that changing the particle count does not reshuffle
    # the diffusion draw, and vice versa.
    seed_stream, diffusion_stream = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.Generator(np.random.PCG64(diffusion_stream))

    lonlat = seed_particles(polygon, n_particles, int(seed_stream.generate_state(1)[0]))
    centroid = polygon.centroid
    frame = Frame.for_point(float(centroid.x), float(centroid.y))
    positions = frame.forward(lonlat[:, 0], lonlat[:, 1])
    seed_positions = positions.copy()

    sign = -1 if mode == "backward" else 1
    t_start_s = t0.timestamp()
    n_steps = int(round(horizon_h * 3600 / dt_s))
    steps_per_snapshot = snapshot_every_s // dt_s

    states = [
        DriftState(
            t_offset_min=0,
            positions=seed_positions.copy(),
            stretch_factor=1.0,
            stretch_factor_raw=1.0,
        )
    ]

    for step in range(1, n_steps + 1):
        t_s = t_start_s + sign * (step - 1) * dt_s
        positions = rk4_step(
            positions, t_s, dt_s, fields, sign, windage, theta_dev_deg, frame
        )
        positions = diffusion_step(positions, k_h, dt_s, rng)

        if step % steps_per_snapshot == 0:
            offset_min = sign * (step * dt_s) // 60
            reported, raw = stretch_factor(
                positions, seed_positions, mode, k_h=k_h, elapsed_s=step * dt_s
            )
            states.append(
                DriftState(
                    t_offset_min=int(offset_min),
                    positions=positions.copy(),
                    stretch_factor=reported,
                    stretch_factor_raw=raw,
                )
            )

    elapsed_s = (datetime.now(tz=UTC) - started).total_seconds()
    logger.info(
        "drift.solver: %s run, %d particles, %d h, field=%s, %d snapshots in %.2f s",
        mode,
        n_particles,
        horizon_h,
        fields.source,
        len(states),
        elapsed_s,
    )

    return DriftRun(
        mode=mode,
        n_particles=n_particles,
        horizon_h=horizon_h,
        windage=windage,
        k_h=k_h,
        theta_dev_deg=theta_dev_deg,
        field_source=fields.source,
        states=states,
        t0=t0,
        frame=frame,
        seed=seed,
        elapsed_s=elapsed_s,
    )


def snapshot_time(drift_run: DriftRun, index: int) -> datetime:
    """Absolute time of a snapshot — t0 plus its signed offset."""
    return drift_run.t0 + timedelta(minutes=drift_run.states[index].t_offset_min)
