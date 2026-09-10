"""CFAR / adaptive-threshold dark-formation detector — the always-works fallback.

§9: keep this dependency-light and never let it import torch. It is the reason
the demo survives a missing weights file, a broken CUDA install, or a laptop
that cannot load the U-Net. numpy and scipy only.

The polygonisation below is hand-rolled for exactly that reason. rasterio and
scikit-image both ship a mask-to-shapes routine and both are already pinned in
requirements.txt, but importing either here would put this module one transitive
edge away from a stack it exists to be independent of. Run-length encoding each
labelled component into per-row rectangles and unioning them costs a few lines
and keeps the dependency set at numpy + scipy + shapely. A test asserts it.

Detection is stage 1 of 4 and the least interesting part of the system (§1). This
module is deliberately unclever: it finds dark formations, and
`discriminator.py` decides whether they are oil.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon

from app.config import settings

logger = logging.getLogger(__name__)

# A geotransform whose pixel side is smaller than this is being handed to us in
# degrees, not metres. §8 forbids computing distances in degrees, so this is an
# error rather than something to correct silently: at Baltic latitudes a degree
# of longitude is ~64 km and a degree of latitude ~111 km, so no single scale
# factor repairs it. The caller reprojects.
MIN_PLAUSIBLE_PIXEL_SIDE_M = 0.05

DEGREES_TRANSFORM = (
    "transform appears to be in degrees (pixel side {side:.2e} < "
    f"{MIN_PLAUSIBLE_PIXEL_SIDE_M} m); reproject the raster to a local UTM zone "
    "before detecting — §8 forbids metric computation in degrees"
)


def _affine(transform: object) -> tuple[float, float, float, float, float, float]:
    """Normalise a rasterio `Affine` or a 6-tuple to (a, b, c, d, e, f).

    Accepting both keeps this module free of a rasterio import while still
    taking what `rasterio.open(...).transform` hands back, since an Affine is
    itself a 6-element sequence in that order.
    """
    if hasattr(transform, "a"):
        return tuple(  # type: ignore[return-value]
            float(getattr(transform, name)) for name in ("a", "b", "c", "d", "e", "f")
        )
    if isinstance(transform, Sequence) and len(transform) >= 6:
        return tuple(float(value) for value in tuple(transform)[:6])  # type: ignore[return-value]
    raise TypeError(
        f"transform must be a rasterio Affine or a 6-tuple, got {type(transform).__name__}"
    )


def pixel_area_m2(transform: object) -> float:
    """Area of one pixel, from the determinant of the affine's linear part."""
    a, b, _, d, e, _ = _affine(transform)
    area = abs(a * e - b * d)
    side = float(np.sqrt(area))
    if side < MIN_PLAUSIBLE_PIXEL_SIDE_M:
        raise ValueError(DEGREES_TRANSFORM.format(side=side))
    return area


def _pixel_to_world(transform: object, geometry: Polygon) -> Polygon:
    """Map a polygon from pixel coordinates into the raster's CRS.

    `shapely.transform` carries interior rings through, so a slick with a hole in
    it stays a slick with a hole in it.
    """
    a, b, c, d, e, f = _affine(transform)

    def apply(coords: np.ndarray) -> np.ndarray:
        cols, rows = coords[:, 0], coords[:, 1]
        return np.column_stack([a * cols + b * rows + c, d * cols + e * rows + f])

    return shapely.transform(geometry, apply)


def adaptive_threshold(
    sigma0_db: np.ndarray,
    window_px: int,
    offset_db: float,
) -> np.ndarray:
    """Local-mean CFAR: flag pixels sitting `offset_db` below the local mean.

    The constant-false-alarm-rate idea is that the decision is made against the
    neighbourhood rather than against a global level, so a scene whose
    brightness varies with incidence angle across the swath does not need
    radiometric flattening first.

    Windows whose local standard deviation is below the noise floor are
    suppressed: they carry no contrast to threshold, and a threshold applied to
    them turns speckle into detections at exactly the rate the window is wide.
    """
    data = np.asarray(sigma0_db, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"sigma0_db must be 2-D, got shape {data.shape}")
    if window_px < 3:
        raise ValueError(f"window_px must be at least 3, got {window_px}")

    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)

    # Normalising by the count of finite pixels rather than the window size keeps
    # the background estimate honest at the swath edge and across nodata.
    weight = ndimage.uniform_filter(finite.astype(np.float64), size=window_px, mode="nearest")
    local_sum = ndimage.uniform_filter(filled, size=window_px, mode="nearest")
    local_sqsum = ndimage.uniform_filter(filled**2, size=window_px, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        local_mean = local_sum / weight
        local_var = local_sqsum / weight - local_mean**2
    local_std = np.sqrt(np.clip(local_var, 0.0, None))

    dark = finite & (data < local_mean - offset_db)
    return dark & (local_std >= settings.cfar_min_local_std_db)


def _clean(mask: np.ndarray, open_px: int, close_px: int) -> np.ndarray:
    """Opening drops speckle; closing seals the pinholes it leaves behind."""
    cleaned = mask
    if open_px > 0:
        cleaned = ndimage.binary_opening(cleaned, structure=np.ones((open_px, open_px)))
    if close_px > 0:
        cleaned = ndimage.binary_closing(cleaned, structure=np.ones((close_px, close_px)))
    return cleaned


def _component_polygon(component: np.ndarray, offset_row: int, offset_col: int) -> Polygon | None:
    """One labelled component to a polygon, via per-row pixel runs.

    Each row of the component becomes a handful of rectangles rather than one box
    per pixel, so a component of N pixels costs O(runs), not O(N). The union of
    those rectangles is the component's exact footprint in pixel coordinates.
    """
    boxes = []
    for row_index, row in enumerate(component):
        if not row.any():
            continue
        # Runs of True: pad with False so a run touching either end still shows
        # up as a transition in the diff.
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in zip(edges[0::2], edges[1::2], strict=True):
            boxes.append(
                shapely.box(
                    offset_col + start,
                    offset_row + row_index,
                    offset_col + stop,
                    offset_row + row_index + 1,
                )
            )
    if not boxes:
        return None
    merged = shapely.union_all(boxes)
    if isinstance(merged, Polygon):
        return merged
    # A component is 4-connected, so the union is a single polygon; a
    # MultiPolygon here means diagonal-only touching survived the morphology.
    # Take the largest part rather than dropping the component.
    return max(merged.geoms, key=lambda part: part.area)


def detect(
    sigma0_db: np.ndarray,
    transform: object,
    min_area_km2: float,
) -> list[Polygon]:
    """Dark formations in a Sigma0 dB scene, as polygons in the raster's CRS.

    The returned polygons are in the coordinates the transform maps to, which
    §8 requires to be a projected metric CRS — `pixel_area_m2` raises on a
    geographic one rather than measuring area in degrees.
    """
    area_per_pixel_m2 = pixel_area_m2(transform)
    min_area_px = (min_area_km2 * 1.0e6) / area_per_pixel_m2

    mask = adaptive_threshold(sigma0_db, settings.cfar_window_px, settings.cfar_offset_db)
    mask = _clean(mask, settings.cfar_open_px, settings.cfar_close_px)

    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        logger.info("classical detector found no dark formations")
        return []

    polygons: list[Polygon] = []
    for label_index, bounds in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = bounds
        component = labels[bounds] == label_index
        if component.sum() < min_area_px:
            continue
        polygon = _component_polygon(component, rows.start, cols.start)
        if polygon is None:
            continue

        world = _pixel_to_world(transform, polygon)
        if settings.detection_simplify_m > 0.0:
            world = world.simplify(settings.detection_simplify_m, preserve_topology=True)
        if world.is_valid and world.area >= min_area_km2 * 1.0e6:
            polygons.append(world)

    polygons.sort(key=lambda item: item.area, reverse=True)
    logger.info(
        "classical detector found %d dark formation(s) above %.2f km2 in %d component(s)",
        len(polygons),
        min_area_km2,
        n_labels,
    )
    return polygons
