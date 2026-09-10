"""Per-polygon geometry and texture features — §5.2.

The 18 features are the exact column set in `detections` (§6). Metric quantities
are computed in a local UTM projection, never in degrees (§8).
"""

from typing import TypedDict

import numpy as np
from shapely.geometry import Polygon

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


class DetectionFeatures(TypedDict):
    area_km2: float
    perimeter_km: float
    shape_complexity: float
    major_axis_km: float
    minor_axis_km: float
    eccentricity: float
    orientation_deg: float
    mean_sigma0_db: float
    std_sigma0_db: float
    contrast_db: float
    edge_gradient_mean: float
    edge_gradient_std: float
    glcm_homogeneity: float
    glcm_contrast: float
    glcm_entropy: float
    wind_speed_ms: float
    distance_to_coast_km: float
    n_ships_within_20km: int


def geometry_features(polygon: Polygon) -> dict[str, float]:
    """Shape complexity is P^2 / 4*pi*A — 1.0 for a circle, higher for a slick."""
    ...


def radiometric_features(
    sigma0_db: np.ndarray,
    polygon: Polygon,
    transform: object,
) -> dict[str, float]:
    """contrast_db is slick mean minus local background mean."""
    ...


def texture_features(
    sigma0_db: np.ndarray,
    polygon: Polygon,
    transform: object,
) -> dict[str, float]: ...


def context_features(
    polygon: Polygon,
    wind_speed_ms: float,
    acq_time: object,
) -> dict[str, float]: ...


def wind_gate_violated(wind_speed_ms: float) -> bool:
    """Below 3 m/s the sea surface itself mimics oil; above 12 m/s slicks
    disperse below detectability. Either way the detection is not trustworthy
    and the UI must say so (§5.2)."""
    ...


def extract(
    polygon: Polygon,
    sigma0_db: np.ndarray,
    transform: object,
    wind_speed_ms: float,
    acq_time: object,
) -> DetectionFeatures: ...
