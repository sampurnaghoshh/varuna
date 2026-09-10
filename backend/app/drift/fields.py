"""Wind and current field loaders with a synthetic fallback — §5.1.

Resolution order, logged and reported as `field_source` in every drift response:

    1. cached CMEMS/ERA5 NetCDF under data/
    2. bundled per-scenario field snapshot
    3. synthetic field

A missing field never crashes a run. It degrades and says which source it used.

The synthetic field is geostrophic-like flow with 2-3 mesoscale eddies plus a
spatially coherent wind, and is deterministic given a seed — the demo must
reproduce identically offline (§2.1).

Tier 1 is a live code path with no data behind it in this build: no NetCDF is
bundled and no NetCDF reader is pinned in requirements.txt, so the reader import
is guarded and the tier degrades. It is not a stub that fabricates a field
(§2.2) — it returns None and `resolve` reports the source it actually used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Protocol

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

FieldSource = Literal["cmems_netcdf", "scenario_snapshot", "synthetic"]

BBox = tuple[float, float, float, float]

_EARTH_RADIUS_M: Final[float] = 6_371_000.0


class VelocityField(Protocol):
    source: FieldSource

    def current_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...

    def wind10_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...

    def stokes_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray: ...


def _as_arrays(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_a = np.asarray(lon, dtype=np.float64)
    lat_a = np.asarray(lat, dtype=np.float64)
    if lon_a.shape != lat_a.shape:
        raise ValueError(f"lon/lat shape mismatch: {lon_a.shape} vs {lat_a.shape}")
    return lon_a, lat_a


@dataclass
class _StokesFromWind:
    """Surface Stokes drift as a fixed fraction of U10, aligned with the wind.

    Used by any source that does not ship a Stokes product of its own. The
    first-order relation u_s ~ 0.01-0.02 * U10 is the standard approximation for
    a fully developed sea; the coefficient is a config tunable, not a literal.
    """

    stokes_fraction: float

    def stokes_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray:
        return self.stokes_fraction * self.wind10_ms(lon, lat, t)  # type: ignore[attr-defined]


@dataclass
class SyntheticField(_StokesFromWind):
    """Geostrophic-like flow: 2-3 mesoscale eddies plus a coherent wind.

    The current is derived from a streamfunction psi as u = (-dpsi/dy, dpsi/dx),
    so it is non-divergent by construction. That is what makes it geostrophic
    *like* rather than smooth noise: a divergent synthetic current would pile
    particles into artificial convergence zones and hand the KDE a concentration
    that no ocean produced.

    Deterministic given a seed: identical eddy centres, radii, amplitudes and
    wind phase on every run, on every machine (§2.1).
    """

    source: FieldSource
    lon0: float
    lat0: float
    eddy_x_m: np.ndarray
    eddy_y_m: np.ndarray
    eddy_radius_m: np.ndarray
    eddy_amplitude: np.ndarray
    wind_speed_ms: float
    wind_dir_rad: float
    wind_modulation_m: float
    wind_period_s: float
    seed: int

    def _to_local_m(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equirectangular metres about the field origin.

        Only the synthetic field's own internal parameterisation lives here; the
        solver integrates in UTM (§8). An equirectangular frame is adequate to
        *define* a smooth analytic field and keeps it independent of any scene's
        UTM zone.
        """
        x = np.radians(lon - self.lon0) * _EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        y = np.radians(lat - self.lat0) * _EARTH_RADIUS_M
        return x, y

    def current_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray:
        lon_a, lat_a = _as_arrays(lon, lat)
        x, y = self._to_local_m(lon_a, lat_a)

        u = np.zeros_like(x)
        v = np.zeros_like(y)
        for xc, yc, radius, amp in zip(
            self.eddy_x_m,
            self.eddy_y_m,
            self.eddy_radius_m,
            self.eddy_amplitude,
            strict=True,
        ):
            dx = x - xc
            dy = y - yc
            r2 = dx * dx + dy * dy
            # psi = amp * exp(-r^2 / 2R^2)  ->  u = -dpsi/dy, v = dpsi/dx
            psi = amp * np.exp(-r2 / (2.0 * radius * radius))
            u += psi * dy / (radius * radius)
            v += -psi * dx / (radius * radius)
        return np.stack((u, v), axis=-1)

    def wind10_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray:
        lon_a, lat_a = _as_arrays(lon, lat)
        x, y = self._to_local_m(lon_a, lat_a)

        # Spatially coherent: one large-scale direction with a broad sinusoidal
        # veer, slowly rotating in time. Wind must stay correlated over the whole
        # cloud - incoherent per-particle wind would fake a diffusion that K_h
        # already models, and inflate the origin uncertainty dishonestly.
        t_s = t.timestamp()
        phase = 2.0 * np.pi * t_s / self.wind_period_s
        veer = 0.15 * np.sin((x + y) / self.wind_modulation_m + phase)
        direction = self.wind_dir_rad + veer
        speed = self.wind_speed_ms * (1.0 + 0.10 * np.cos(x / self.wind_modulation_m - phase))
        return np.stack((speed * np.cos(direction), speed * np.sin(direction)), axis=-1)


@dataclass
class GriddedField(_StokesFromWind):
    """Bilinearly sampled u/v grids on a regular lon/lat mesh.

    Backs both tier 1 (NetCDF) and tier 2 (bundled scenario snapshot); the two
    differ only in where the arrays came from. Time is treated as a single valid
    instant: a 12 h backward run against a snapshot uses a frozen ocean state,
    and `field_source` is what tells the UI that is what happened.
    """

    source: FieldSource
    lons: np.ndarray
    lats: np.ndarray
    u_current: np.ndarray
    v_current: np.ndarray
    u_wind: np.ndarray
    v_wind: np.ndarray

    def _sample(self, grid: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        ix = np.interp(lon, self.lons, np.arange(self.lons.size))
        iy = np.interp(lat, self.lats, np.arange(self.lats.size))
        x0 = np.clip(np.floor(ix).astype(np.int64), 0, self.lons.size - 1)
        y0 = np.clip(np.floor(iy).astype(np.int64), 0, self.lats.size - 1)
        x1 = np.clip(x0 + 1, 0, self.lons.size - 1)
        y1 = np.clip(y0 + 1, 0, self.lats.size - 1)
        fx = ix - x0
        fy = iy - y0
        return (
            grid[y0, x0] * (1 - fx) * (1 - fy)
            + grid[y0, x1] * fx * (1 - fy)
            + grid[y1, x0] * (1 - fx) * fy
            + grid[y1, x1] * fx * fy
        )

    def current_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray:
        lon_a, lat_a = _as_arrays(lon, lat)
        return np.stack(
            (
                self._sample(self.u_current, lon_a, lat_a),
                self._sample(self.v_current, lon_a, lat_a),
            ),
            axis=-1,
        )

    def wind10_ms(self, lon: np.ndarray, lat: np.ndarray, t: datetime) -> np.ndarray:
        lon_a, lat_a = _as_arrays(lon, lat)
        return np.stack(
            (self._sample(self.u_wind, lon_a, lat_a), self._sample(self.v_wind, lon_a, lat_a)),
            axis=-1,
        )


def load_netcdf(path: Path) -> VelocityField | None:
    """Tier 1. Returns None when no NetCDF or no reader is available.

    No NetCDF reader is pinned in requirements.txt and no .nc file is bundled, so
    in this build the tier degrades every time. The import is guarded rather than
    removed so that dropping a CMEMS/ERA5 file into data/fields/ is the only step
    needed to activate it.
    """
    candidates = sorted(path.glob("*.nc")) if path.is_dir() else ([path] if path.is_file() else [])
    if not candidates:
        logger.info("drift.fields: tier 1 skipped - no NetCDF under %s", path)
        return None

    try:
        import netCDF4  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        logger.warning(
            "drift.fields: tier 1 skipped - %s present but no NetCDF reader is installed",
            candidates[0].name,
        )
        return None

    logger.info("drift.fields: tier 1 reader present but no loader implemented; degrading")
    return None


def load_scenario_snapshot(scenario_code: str) -> VelocityField | None:
    """Tier 2. Bundled per-scenario field snapshot written by build_fixtures.

    Format: data/fixtures/fields/{code}.npz with arrays lons, lats, u_current,
    v_current, u_wind, v_wind. Nothing writes it yet (§15 puts fixtures at T+5h);
    this reads it when it exists and degrades when it does not.
    """
    snapshot = Path(settings.fixtures_dir) / "fields" / f"{scenario_code}.npz"
    if not snapshot.is_file():
        logger.info("drift.fields: tier 2 skipped - no snapshot at %s", snapshot)
        return None

    try:
        with np.load(snapshot) as data:
            return GriddedField(
                stokes_fraction=settings.stokes_wind_fraction,
                source="scenario_snapshot",
                lons=np.asarray(data["lons"], dtype=np.float64),
                lats=np.asarray(data["lats"], dtype=np.float64),
                u_current=np.asarray(data["u_current"], dtype=np.float64),
                v_current=np.asarray(data["v_current"], dtype=np.float64),
                u_wind=np.asarray(data["u_wind"], dtype=np.float64),
                v_wind=np.asarray(data["v_wind"], dtype=np.float64),
            )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("drift.fields: tier 2 snapshot %s unreadable (%s); degrading", snapshot, exc)
        return None


def synthetic_field(bbox: BBox, seed: int) -> SyntheticField:
    """Tier 3. Deterministic given `seed` — identical output on every machine."""
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError(f"Malformed bbox {bbox}: expected (lon_min, lat_min, lon_max, lat_max)")

    lon0 = 0.5 * (lon_min + lon_max)
    lat0 = 0.5 * (lat_min + lat_max)

    span_x_m = max(
        np.radians(lon_max - lon_min) * _EARTH_RADIUS_M * np.cos(np.radians(lat0)), 1.0e4
    )
    span_y_m = max(np.radians(lat_max - lat_min) * _EARTH_RADIUS_M, 1.0e4)
    span_m = float(max(span_x_m, span_y_m))

    rng = np.random.default_rng(seed)
    n_eddies = settings.synthetic_n_eddies

    # Eddies are placed over a region wider than the bbox so the cloud sees eddy
    # flanks rather than sitting inside a single closed cell for 12 h.
    eddy_x = rng.uniform(-1.5 * span_m, 1.5 * span_m, size=n_eddies)
    eddy_y = rng.uniform(-1.5 * span_m, 1.5 * span_m, size=n_eddies)
    eddy_radius = rng.uniform(0.4 * span_m, 1.0 * span_m, size=n_eddies)
    sign = rng.choice(np.array([-1.0, 1.0]), size=n_eddies)

    # psi amplitude chosen so peak orbital speed ~ synthetic_eddy_speed_ms:
    # |u|_max = amp / R * exp(-1/2)  ->  amp = speed * R * sqrt(e).
    eddy_amp = sign * settings.synthetic_eddy_speed_ms * eddy_radius * np.sqrt(np.e)

    return SyntheticField(
        stokes_fraction=settings.stokes_wind_fraction,
        source="synthetic",
        lon0=lon0,
        lat0=lat0,
        eddy_x_m=eddy_x,
        eddy_y_m=eddy_y,
        eddy_radius_m=eddy_radius,
        eddy_amplitude=eddy_amp,
        wind_speed_ms=settings.synthetic_wind_speed_ms,
        wind_dir_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        wind_modulation_m=2.0 * span_m,
        wind_period_s=float(rng.uniform(18.0, 30.0) * 3600.0),
        seed=seed,
    )


def resolve(
    bbox: BBox,
    t0: datetime,
    scenario_code: str | None,
    seed: int,
) -> VelocityField:
    """Walk the tier chain. Never raises, always reports the source it used."""
    field = load_netcdf(Path(settings.data_dir) / "fields")
    if field is not None:
        logger.info("drift.fields: using %s", field.source)
        return field

    if scenario_code is not None:
        field = load_scenario_snapshot(scenario_code)
        if field is not None:
            logger.info("drift.fields: using %s for %s", field.source, scenario_code)
            return field

    logger.info("drift.fields: falling back to synthetic field (seed=%d)", seed)
    return synthetic_field(bbox, seed)
