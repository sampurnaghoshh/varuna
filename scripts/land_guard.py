"""Refuse to write a scenario whose oil is sitting on dry land.

`make fixtures` (§13) once placed SC-01 over Bornholm: 80 of 116 slick vertices,
the authored release point and ~90% of the particles at the worst snapshot were
on the island. Every number downstream — the drift run, the recovery error, the
LR — was arithmetically correct and physically meaningless, and nothing in the
pipeline noticed, because nothing in the pipeline had an opinion about land.

This module is that opinion. It is imported by `scripts/build_fixtures.py` and
raises `LandPlacementError`; it does not warn, and it does not filter. A warning
scrolls past during a 3-minute regeneration and a filter silently changes the
scenario into a different one. The build stops.

Same shape and same reason as `scripts/data_split.py`: one implementation of a
rule that must not be restated anywhere, enforced programmatically rather than
by convention, raising rather than degrading.

WHAT IS CHECKED, AND WHAT DELIBERATELY IS NOT
---------------------------------------------
Checked: slick and dark-formation vertices, the authored release point and
corridor, and the particle cloud at every drift snapshot.

Not checked: the scene footprint. A real Sentinel-1 scene contains coastline —
that is normal and is half of why SAR oil detection is hard. A guard that
refused a scene overlapping land would refuse every real scene we ever ingest.

The particle threshold is a fraction, not zero, because a 12 h cloud legitimately
has tails: a handful of particles grounding at the far edge of the uncertainty
hull is a physical statement about where the oil might have come from. Most of
the cloud on land is not a statement, it is a misplaced scenario.

THE MASK
--------
`data/geo/baltic_land_mask.geojson`, written by `scripts/fetch_coastline.py`.
NOT `baltic_land.geojson` — that one is the console basemap, where every
landmass crossing the AOI edge is a stroked LineString rather than a polygon, so
a point-in-polygon test against it reports the Polish coast as open water.

Geometry outside the mask's declared AOI raises too. "Off the edge of the mask"
is not "not on land", and a guard that quietly conflated the two would pass
exactly the placements it exists to catch.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

MASK_FILENAME = "baltic_land_mask.geojson"
REPO_ROOT = Path(__file__).resolve().parent.parent

# A 12 h particle cloud is allowed tails that grounding is a real answer for; it
# is not allowed to be mostly ashore. Zero would reject physically honest runs,
# and anything much larger stops distinguishing a tail from a misplacement.
MAX_PARTICLE_LAND_FRACTION = 0.05

# A slick vertex or a release point has no tail to allow. Either the scenario is
# over water or it is not.
NO_TOLERANCE = 0.0


class LandPlacementError(RuntimeError):
    """Raised when scenario geometry is on land, or outside the mask's AOI.

    Deliberately not a subclass of anything `build_fixtures.py` catches. The
    build stops, nothing is written, and a human decides where the scenario
    should actually go.
    """


class LandMaskUnavailableError(RuntimeError):
    """Raised when the mask is missing.

    Also fatal, and for the same reason: a guard that skips itself when its
    reference data is absent is not a guard. Run
    `docker compose exec -T backend python scripts/fetch_coastline.py`.
    """


@dataclass(frozen=True)
class PlacementReport:
    """What the guard measured for one piece of geometry.

    Returned on success so the numbers can be recorded in the fixture set
    instead of merely having been checked — a clearance figure nobody can read
    is indistinguishable from a guard that never ran.
    """

    label: str
    n_checked: int
    n_on_land: int
    fraction_on_land: float
    clearance_km: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_checked": self.n_checked,
            "n_on_land": self.n_on_land,
            "fraction_on_land": round(self.fraction_on_land, 6),
            "clearance_km": (
                None if self.clearance_km is None else round(self.clearance_km, 3)
            ),
        }


@dataclass(frozen=True)
class LandMask:
    """Baltic land as closed polygons, plus the AOI it is valid inside."""

    land: BaseGeometry
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    source: str

    def on_land(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Vectorised point-in-polygon, in degrees.

        Containment is topological, so it is correct in geographic coordinates —
        §8's "never compute distances in degrees" is about metric quantities,
        and this is not one. `clearance_km` does project.
        """
        return shapely.contains_xy(self.land, np.asarray(lon), np.asarray(lat))

    def outside_aoi(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        return (
            (lon < self.lon_min)
            | (lon > self.lon_max)
            | (lat < self.lat_min)
            | (lat > self.lat_max)
        )

    def clearance_km(self, geometry_wgs84: BaseGeometry) -> float:
        """Distance from geometry to the nearest land, in km. 0.0 if touching.

        Projected to the geometry's own UTM zone first (§8). Returns 0.0 rather
        than a negative for geometry that intersects land — the caller has
        already raised by then, and a signed distance would invite someone to
        treat "how far ashore" as a tolerance.
        """
        from pyproj import CRS, Transformer
        from shapely.ops import transform as shapely_transform

        centroid = geometry_wgs84.centroid
        zone = int((centroid.x + 180.0) // 6.0) % 60 + 1
        epsg = (32600 if centroid.y >= 0.0 else 32700) + zone
        to_utm = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True
        )

        def project(geometry: BaseGeometry) -> BaseGeometry:
            return shapely_transform(lambda x, y: to_utm.transform(x, y), geometry)

        target = project(geometry_wgs84)
        land = project(self.land)
        if target.intersects(land):
            return 0.0
        return float(land.distance(target)) / 1000.0


def mask_path() -> Path:
    """Where the mask lives, on the host and in the container alike."""
    override = os.environ.get("VARUNA_GEO_DIR")
    if override:
        candidate = Path(override) / MASK_FILENAME
        if candidate.is_file():
            return candidate
    return REPO_ROOT / "data" / "geo" / MASK_FILENAME


def load_mask(path: Path | None = None) -> LandMask:
    """Read the land mask, or raise. Never returns a degraded stand-in."""
    path = mask_path() if path is None else Path(path)
    if not path.is_file():
        raise LandMaskUnavailableError(
            f"No land mask at {path}. The placement guard cannot run without it; "
            "regenerate with `docker compose exec -T backend python "
            "scripts/fetch_coastline.py`."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    properties = payload.get("properties", {})
    aoi = properties.get("aoi", {})
    lon = aoi.get("lon")
    lat = aoi.get("lat")
    if not lon or not lat:
        raise LandMaskUnavailableError(
            f"Land mask at {path} declares no AOI, so geometry outside it could "
            "not be distinguished from geometry over water."
        )

    land = shape(payload["geometry"])
    shapely.prepare(land)
    return LandMask(
        land=land,
        lon_min=float(lon[0]),
        lon_max=float(lon[1]),
        lat_min=float(lat[0]),
        lat_max=float(lat[1]),
        source=str(properties.get("source", path.name)),
    )


def check_points(
    mask: LandMask,
    label: str,
    lon: np.ndarray,
    lat: np.ndarray,
    max_fraction_on_land: float = NO_TOLERANCE,
    clearance_of: BaseGeometry | None = None,
) -> PlacementReport:
    """Assert that points are over water, or raise `LandPlacementError`.

    `max_fraction_on_land` is the only knob, and it exists for particle clouds
    alone (see `MAX_PARTICLE_LAND_FRACTION`). Everything else passes the default
    of zero.
    """
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    if lon.size == 0:
        raise LandPlacementError(f"{label}: nothing to check; refusing to pass by default.")

    outside = mask.outside_aoi(lon, lat)
    if outside.any():
        index = int(np.argmax(outside))
        raise LandPlacementError(
            f"{label}: {int(outside.sum())} of {lon.size} points fall outside the land "
            f"mask AOI ({mask.lon_min}–{mask.lon_max} E, {mask.lat_min}–{mask.lat_max} N) "
            f"— first at {lon[index]:.4f}, {lat[index]:.4f}. Outside the mask is not "
            "water, it is unknown, so this cannot be passed. Widen the mask AOI in "
            "scripts/fetch_coastline.py or move the scenario."
        )

    hits = mask.on_land(lon, lat)
    n_on_land = int(hits.sum())
    fraction = n_on_land / lon.size

    if fraction > max_fraction_on_land:
        index = int(np.argmax(hits))
        allowance = (
            "no point may be on land"
            if max_fraction_on_land <= 0.0
            else f"at most {max_fraction_on_land:.1%} may be"
        )
        raise LandPlacementError(
            f"{label}: {n_on_land} of {lon.size} points ({fraction:.1%}) are on land "
            f"— {allowance}. First offender at {lon[index]:.4f}, {lat[index]:.4f}. "
            "The scenario is misplaced; move it over open water rather than "
            "relaxing this guard."
        )

    clearance = None if clearance_of is None else mask.clearance_km(clearance_of)
    report = PlacementReport(
        label=label,
        n_checked=int(lon.size),
        n_on_land=n_on_land,
        fraction_on_land=fraction,
        clearance_km=clearance,
    )
    logger.info(
        "land guard: %s over water (%d points%s)",
        label,
        report.n_checked,
        "" if clearance is None else f", {clearance:.1f} km clearance",
    )
    return report


def check_polygon(mask: LandMask, label: str, polygon_wgs84: BaseGeometry) -> PlacementReport:
    """Every vertex of a polygon, plus its clearance to the nearest land."""
    coords = np.asarray(shapely.get_coordinates(polygon_wgs84), dtype=np.float64)
    return check_points(
        mask, label, coords[:, 0], coords[:, 1], clearance_of=polygon_wgs84
    )
