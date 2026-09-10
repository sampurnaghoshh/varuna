"""Per-polygon geometry and texture features — §5.2.

The 18 features are the exact column set in `detections` (§6). Metric quantities
are computed in a local UTM projection, never in degrees (§8).

Absence is a first-class outcome here. There is no SAR raster in this build, so
the eight radiometric and texture features have nothing to measure, and there is
no bundled coastline, so `distance_to_coast_km` has nothing to measure against.
Those come back as None with a stated reason rather than as a zero, because a
zero contrast reads like a measurement of no damping and a zero distance to
coast reads like a beached slick. `extract` returns the reasons alongside the
values, and every consumer — the discriminator, the explainability payload, the
fixture generator — carries them through to the UI (§2.2).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
from shapely.geometry import Polygon

from app.config import settings

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "area_km2",
    "perimeter_km",
    "shape_complexity",
    "major_axis_km",
    "minor_axis_km",
    "eccentricity",
    "orientation_deg",
    "mean_sigma0_db",
    "std_sigma0_db",
    "contrast_db",
    "edge_gradient_mean",
    "edge_gradient_std",
    "glcm_homogeneity",
    "glcm_contrast",
    "glcm_entropy",
    "wind_speed_ms",
    "distance_to_coast_km",
    "n_ships_within_20km",
)

GEOMETRY_FEATURES: tuple[str, ...] = FEATURE_NAMES[:7]
RADIOMETRIC_FEATURES: tuple[str, ...] = FEATURE_NAMES[7:12]
TEXTURE_FEATURES: tuple[str, ...] = FEATURE_NAMES[12:15]

# Reasons travel with the value into the explainability payload verbatim (§7),
# and match the strings the fixture set already carries.
NO_PIXELS = "no SAR raster in this build; radiometric and texture features need pixels"
NO_COASTLINE = "no coastline dataset bundled in this build"
NO_AIS_FRAME = "no AIS frame bundled with this scenario"

# §5.2 wind gate, stated once. The UI renders this next to the flag: it is one of
# the strongest answers we have in Q&A, and it only lands if the reason is on
# screen rather than just the boolean.
WIND_GATE_LOW = (
    "wind {wind:.1f} m/s is below the {floor:.1f} m/s gate: at that speed the sea "
    "surface itself mimics oil, so a dark formation is not trustworthy evidence "
    "of a slick"
)
WIND_GATE_HIGH = (
    "wind {wind:.1f} m/s is above the {ceiling:.1f} m/s gate: slicks disperse "
    "below detectability at that speed, so a dark formation is unlikely to still "
    "be oil"
)
WIND_GATE_OK = "wind {wind:.1f} m/s is inside the {floor:.1f}-{ceiling:.1f} m/s gate"
NO_WIND = "no wind speed for this scene; the §5.2 gate cannot be evaluated"

SHIP_RADIUS_KM = 20.0
GLCM_LEVELS = 32


class DetectionFeatures(TypedDict):
    area_km2: float | None
    perimeter_km: float | None
    shape_complexity: float | None
    major_axis_km: float | None
    minor_axis_km: float | None
    eccentricity: float | None
    orientation_deg: float | None
    mean_sigma0_db: float | None
    std_sigma0_db: float | None
    contrast_db: float | None
    edge_gradient_mean: float | None
    edge_gradient_std: float | None
    glcm_homogeneity: float | None
    glcm_contrast: float | None
    glcm_entropy: float | None
    wind_speed_ms: float | None
    distance_to_coast_km: float | None
    n_ships_within_20km: int | None


@dataclass(frozen=True)
class ExtractedFeatures:
    """The §5.2 feature vector plus why anything in it is missing.

    Mirrors `attribution.channels.ChannelScores`: a value and, where there is no
    value, the reason in words rather than a substitute number.
    """

    values: DetectionFeatures
    unavailable: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        return self.values.get(name)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(name for name in FEATURE_NAMES if self.values.get(name) is not None)


def _bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing of a vector in a metric frame, folded to [0, 180).

    An axis has no head or tail, so a slick oriented 200 deg and one oriented
    20 deg are the same orientation. E2 folds the same way (§5.3).
    """
    return math.degrees(math.atan2(dx, dy)) % 180.0


def geometry_features(polygon: Polygon) -> dict[str, float]:
    """Shape complexity is P^2 / 4*pi*A — 1.0 for a circle, higher for a slick.

    Axes come from the minimum rotated rectangle rather than from a PCA of the
    boundary: a drifting slick is often crescent-shaped or forked, and the second
    moment of such a shape understates its true extent, which is the quantity
    E3 divides by to recover the released patch length (§5.3).

    The polygon must be in a metric frame (§8) — the caller projects.
    """
    area_m2 = float(polygon.area)
    perimeter_m = float(polygon.length)
    if area_m2 <= 0.0:
        raise ValueError("polygon has no area; cannot extract geometry features")

    corners = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords[:4])
    edge_a = corners[1] - corners[0]
    edge_b = corners[2] - corners[1]
    length_a = float(np.hypot(*edge_a))
    length_b = float(np.hypot(*edge_b))
    major_edge, major_m, minor_m = (
        (edge_a, length_a, length_b) if length_a >= length_b else (edge_b, length_b, length_a)
    )
    axis_ratio = minor_m / major_m if major_m > 0.0 else 0.0

    return {
        "area_km2": area_m2 / 1.0e6,
        "perimeter_km": perimeter_m / 1.0e3,
        "shape_complexity": perimeter_m**2 / (4.0 * math.pi * area_m2),
        "major_axis_km": major_m / 1.0e3,
        "minor_axis_km": minor_m / 1.0e3,
        "eccentricity": math.sqrt(max(1.0 - axis_ratio**2, 0.0)),
        "orientation_deg": _bearing_deg(float(major_edge[0]), float(major_edge[1])),
    }


def _rasterise(polygon: Polygon, shape: tuple[int, int], transform: object) -> np.ndarray:
    """Polygon to a boolean pixel mask.

    rasterio is imported here rather than at module scope: `classical.py` is the
    §9 always-works path and must stay clear of it, and a caller that only wants
    geometry features should not pay for a GDAL import.
    """
    from rasterio.features import rasterize

    burned = rasterize(
        [(polygon, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8"
    )
    return burned.astype(bool)


def radiometric_features(
    sigma0_db: np.ndarray,
    polygon: Polygon,
    transform: object,
) -> dict[str, float]:
    """contrast_db is slick mean minus local background mean.

    The background is an annulus around the slick rather than the whole scene:
    Sigma0 falls off across the swath with incidence angle, so a scene-wide mean
    would read that gradient as damping wherever the slick sits far-range.
    """
    from scipy import ndimage

    data = np.asarray(sigma0_db, dtype=np.float64)
    mask = _rasterise(polygon, data.shape, transform)
    if not mask.any():
        raise ValueError("polygon does not overlap the raster")

    # The annulus starts one slick-radius out and is one radius thick, so it
    # tracks the feature's own scale instead of a fixed distance.
    radius_m = math.sqrt(polygon.area / math.pi)
    inner = _rasterise(polygon.buffer(radius_m), data.shape, transform)
    outer = _rasterise(polygon.buffer(2.0 * radius_m), data.shape, transform)
    background = outer & ~inner & np.isfinite(data)

    slick = mask & np.isfinite(data)
    slick_values = data[slick]
    if slick_values.size == 0:
        raise ValueError("polygon covers no finite pixels")

    mean_db = float(np.mean(slick_values))
    background_db = float(np.mean(data[background])) if background.any() else mean_db

    gradient_y, gradient_x = np.gradient(np.where(np.isfinite(data), data, 0.0))
    gradient = np.hypot(gradient_x, gradient_y)
    # The boundary, one pixel wide: the mask minus its own erosion.
    edge = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    edge_values = gradient[edge] if edge.any() else gradient[mask]

    return {
        "mean_sigma0_db": mean_db,
        "std_sigma0_db": float(np.std(slick_values)),
        "contrast_db": mean_db - background_db,
        "edge_gradient_mean": float(np.mean(edge_values)),
        "edge_gradient_std": float(np.std(edge_values)),
    }


def texture_features(
    sigma0_db: np.ndarray,
    polygon: Polygon,
    transform: object,
) -> dict[str, float]:
    """GLCM homogeneity, contrast and entropy over the slick's own pixels.

    Texture separates a slick, whose interior is smooth because the damping is
    uniform, from a low-wind cell, whose interior still carries the sea's own
    structure. Quantisation is over the slick's own dynamic range so the measure
    does not depend on where the scene sits in absolute dB.
    """
    from skimage.feature import graycomatrix, graycoprops

    data = np.asarray(sigma0_db, dtype=np.float64)
    mask = _rasterise(polygon, data.shape, transform)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("polygon does not overlap the raster")

    patch = data[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    patch_mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    inside = patch_mask & np.isfinite(patch)
    if inside.sum() < 2:
        raise ValueError("polygon covers too few finite pixels for a GLCM")

    low, high = float(np.min(patch[inside])), float(np.max(patch[inside]))
    if high <= low:
        raise ValueError("polygon pixels have no dynamic range for a GLCM")

    scaled = np.zeros(patch.shape, dtype=np.uint8)
    normalised = (patch[inside] - low) / (high - low)
    scaled[inside] = np.clip((normalised * (GLCM_LEVELS - 1)).round(), 0, GLCM_LEVELS - 1)

    glcm = graycomatrix(
        scaled,
        distances=[1],
        angles=[0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    # graycoprops has no entropy, so it comes off the normalised matrix directly.
    probabilities = glcm[..., 0, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probabilities > 0.0, np.log(probabilities), 0.0)
    entropy = float(np.mean(-np.sum(probabilities * logs, axis=(0, 1))))

    return {
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_entropy": entropy,
    }


def context_features(
    polygon: Polygon,
    wind_speed_ms: float | None,
    acq_time: object = None,
    coastline: Any = None,
    vessel_positions_m: Sequence[tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Wind, coast distance and local traffic — the three non-image features.

    `coastline` and `vessel_positions_m` are optional because neither ships in
    this build. Absent, the feature is None with a reason rather than a zero:
    `distance_to_coast_km = 0` reads as a beached slick and
    `n_ships_within_20km = 0` reads as an empty sea, and both are claims we have
    not measured (§2.2). `acq_time` is carried for the caller's provenance and
    is not itself a feature.
    """
    del acq_time  # provenance only; not one of the 18

    values: dict[str, Any] = {"wind_speed_ms": wind_speed_ms}
    unavailable: dict[str, str] = {}

    if wind_speed_ms is None:
        unavailable["wind_speed_ms"] = NO_WIND

    if coastline is None:
        values["distance_to_coast_km"] = None
        unavailable["distance_to_coast_km"] = NO_COASTLINE
    else:
        values["distance_to_coast_km"] = float(polygon.distance(coastline)) / 1.0e3

    if vessel_positions_m is None:
        values["n_ships_within_20km"] = None
        unavailable["n_ships_within_20km"] = NO_AIS_FRAME
    else:
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])
        positions = np.asarray(vessel_positions_m, dtype=np.float64).reshape(-1, 2)
        distances_m = np.hypot(*(positions - centroid).T) if positions.size else np.empty(0)
        values["n_ships_within_20km"] = int(np.count_nonzero(distances_m <= SHIP_RADIUS_KM * 1.0e3))

    return values, unavailable


def wind_gate_violated(wind_speed_ms: float) -> bool:
    """Below 3 m/s the sea surface itself mimics oil; above 12 m/s slicks
    disperse below detectability. Either way the detection is not trustworthy
    and the UI must say so (§5.2)."""
    return wind_speed_ms < settings.wind_gate_min_ms or wind_speed_ms > settings.wind_gate_max_ms


def wind_gate_reason(wind_speed_ms: float | None) -> str:
    """The gate in words, for the explainability panel (§5.2)."""
    if wind_speed_ms is None:
        return NO_WIND
    if wind_speed_ms < settings.wind_gate_min_ms:
        return WIND_GATE_LOW.format(wind=wind_speed_ms, floor=settings.wind_gate_min_ms)
    if wind_speed_ms > settings.wind_gate_max_ms:
        return WIND_GATE_HIGH.format(wind=wind_speed_ms, ceiling=settings.wind_gate_max_ms)
    return WIND_GATE_OK.format(
        wind=wind_speed_ms, floor=settings.wind_gate_min_ms, ceiling=settings.wind_gate_max_ms
    )


def extract(
    polygon: Polygon,
    sigma0_db: np.ndarray | None = None,
    transform: object = None,
    wind_speed_ms: float | None = None,
    acq_time: object = None,
    coastline: Any = None,
    vessel_positions_m: Sequence[tuple[float, float]] | None = None,
) -> ExtractedFeatures:
    """The §5.2 feature vector for one dark formation, in a metric frame (§8).

    `sigma0_db` is optional: with no raster the eight image-derived features come
    back None with NO_PIXELS, and the geometry, wind and context features are
    still real. That is the current build, and it is a supported path rather
    than a degraded one.
    """
    values: dict[str, Any] = dict.fromkeys(FEATURE_NAMES)
    unavailable: dict[str, str] = {}

    values.update(geometry_features(polygon))

    if sigma0_db is None or transform is None:
        for name in RADIOMETRIC_FEATURES + TEXTURE_FEATURES:
            unavailable[name] = NO_PIXELS
    else:
        for group, compute in (
            (RADIOMETRIC_FEATURES, radiometric_features),
            (TEXTURE_FEATURES, texture_features),
        ):
            try:
                values.update(compute(sigma0_db, polygon, transform))
            except ValueError as error:
                logger.warning("feature group %s unavailable: %s", group[0], error)
                for name in group:
                    unavailable[name] = str(error)

    context_values, context_unavailable = context_features(
        polygon, wind_speed_ms, acq_time, coastline, vessel_positions_m
    )
    values.update(context_values)
    unavailable.update(context_unavailable)

    return ExtractedFeatures(values=values, unavailable=unavailable)  # type: ignore[arg-type]
