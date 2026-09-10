"""Synthetic Sigma0 imagery for the SC-01 demo scene — §0 Tier 4, elected.

WHAT THIS IS, SAID PLAINLY
--------------------------
These are not SAR pixels. They are a physically-argued picture of what the
authored SC-01 slick would look like to Sentinel-1, generated so that the two
§5.2 pixel-derived rule terms — `contrast_db` and `edge_gradient_mean` — have
something real to measure instead of coming back None.

§0 lists synthetic SAR as Tier 4 and requires that it be reached by an explicit
human decision, never as a silent fallback. It was: this module is invoked from
`build_fixtures.py` for SC-01 alone, and everything that surfaces it carries
`SYNTHETIC — NOT SAR IMAGERY`. Nothing degrades into calling it.

It is scenario imagery for a demo. It is NOT evidence that detection works on
real SAR, and no detection metric may be quoted from it (§16).

THE METHOD, NOT A KNOB
----------------------
Every number below traces to something outside this file. If a judge asks how
the image was made, the answer is this list, in order:

1. DARK-REGION GEOMETRY comes from the slick polygon itself — the alpha hull of
   the real 6 h forward drift run. Nothing here draws a shape. The polygon is
   rasterised; that is the entire extent of this module's opinion about where
   the oil is.

2. DAMPING DEPTH is the midpoint of the 3–10 dB range CLAUDE.md states twice for
   mineral oil against the surrounding sea (§5.2 rationale, and the
   `cfar_offset_db` / `rule_w_contrast` notes in config.py). Midpoint, because a
   mid-range slick is the honest default when the scenario does not specify a
   thickness. Fixed before the discriminator was ever run on it and not touched
   afterwards.

   It is deliberately NOT derived from `rule_contrast_pivot_db`. Building the
   image out of the constants that score the image would be circular — the
   scorer would be measuring its own settings, and P(oil) would mean nothing.
   The physical range and the scorer's pivot are independent statements that
   happen to be about the same quantity, and they must stay independent.

3. EDGE SHARPNESS is one resolution cell. An oil boundary is sharp on the water
   — far sharper than the sensor — so what the product shows is the imaging and
   multilooking response, which smears a step over roughly one cell. At this
   product's spacing that is one pixel.

4. SPECKLE is Gamma-distributed intensity, which is what multi-look SAR
   intensity actually is, at an ENL derived from the product chain rather than
   chosen: IW GRDH carries ~4.4 looks at 10 m, and resampling to 50 m averages
   25 pixels, so ENL ~ 4.4 x 25 = 110. That lands near 0.4 dB of speckle in
   dB space, which is why the slick edge is legible at all at this spacing.

5. BACKGROUND LEVEL AND TEXTURE come from the drift run's own wind field. The
   raster samples `field.wind10_ms` on its own grid and converts to Sigma0
   through the power law below, so the image and the physics cannot disagree
   about the wind — the scene's large-scale brightness texture IS the wind field
   the solver integrated.

6. RANGE TREND: VV Sigma0 falls with incidence angle across the swath, ~6 dB
   over the 250 km IW swath, hence the per-km figure below. This is the reason
   `features.radiometric_features` measures contrast against a local annulus
   rather than a scene-wide mean, so the raster has to actually carry it or that
   care would be untested.

The wind-to-Sigma0 relation is a first-order power law, NOT CMOD. Calling it
CMOD would be the same species of claim as calling a rule-based scorer a trained
model (§2.2). It is the standard Sigma0 ~ U^1.2 approximation for VV at moderate
incidence, pinned to a reference point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

BADGE = "SYNTHETIC — NOT SAR IMAGERY"

# Part III product geometry: 2048 x 2048 over the scene footprint (§0). The
# spacing is a consequence of those two, not an independent choice.
SCENE_PX = 2048
SCENE_HALF_SPAN_M = 51_200.0
PIXEL_SPACING_M = 2.0 * SCENE_HALF_SPAN_M / SCENE_PX  # 50.0 m

# Nominal incidence for the reference backscatter level below. Mid-swath IW.
INCIDENCE_ANGLE_DEG = 35.0

# --- 2. damping ---------------------------------------------------------------
# The range CLAUDE.md states for mineral oil against the surrounding sea. The
# authored slick is a mid-range one; the midpoint is the rule, 6.5 dB is its
# consequence. Never adjusted to move P(oil).
OIL_DAMPING_RANGE_DB = (3.0, 10.0)
OIL_DAMPING_DB = 0.5 * (OIL_DAMPING_RANGE_DB[0] + OIL_DAMPING_RANGE_DB[1])

# --- 3. edge ------------------------------------------------------------------
# One resolution cell. The oil boundary is sharp; this is the sensor, not the oil.
EDGE_SMOOTH_CELLS = 1.0

# --- 4. speckle ---------------------------------------------------------------
GRDH_LOOKS = 4.4
GRDH_PIXEL_M = 10.0
EQUIVALENT_NUMBER_OF_LOOKS = GRDH_LOOKS * (PIXEL_SPACING_M / GRDH_PIXEL_M) ** 2  # ~110

# --- 5. wind law --------------------------------------------------------------
# VV Sigma0 ~ U^1.2 at moderate incidence, pinned at -10 dB for 7 m/s at 35 deg.
# First-order approximation, not CMOD.
SIGMA0_REF_DB = -10.0
WIND_REF_MS = 7.0
WIND_EXPONENT = 1.2
WIND_LAW_DB_PER_DECADE = 10.0 * WIND_EXPONENT  # 12 dB per decade of wind speed

# --- 6. range trend -----------------------------------------------------------
# ~6 dB across the 250 km IW swath.
SWATH_FALLOFF_DB = 6.0
SWATH_WIDTH_KM = 250.0
RANGE_TREND_DB_PER_KM = -SWATH_FALLOFF_DB / SWATH_WIDTH_KM  # -0.024 dB/km

# GeoTIFF storage: int16 hundredths of a dB. The quantisation step is 0.01 dB,
# more than an order of magnitude finer than the ~0.4 dB speckle, so it is below
# the noise it stores. Halves the file against float32.
DB_SCALE = 100.0
NODATA_DB = -99.99


@dataclass(frozen=True)
class SyntheticScene:
    """A synthetic Sigma0 raster and the account of how it was made."""

    sigma0_db: np.ndarray
    transform: Any
    crs_epsg: int
    pixel_spacing_m: float
    incidence_angle_deg: float
    method: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.sigma0_db.shape[0]), int(self.sigma0_db.shape[1]))


def sigma0_for_wind(wind_speed_ms: np.ndarray | float) -> np.ndarray | float:
    """VV Sigma0 in dB for a wind speed, by the power law in this module's header.

    Floored at a light-air speed rather than allowed to run to -inf: the law is
    an approximation over the range SAR wind retrieval is valid in, and taking it
    literally at zero wind would invent a black sea.
    """
    speed = np.maximum(np.asarray(wind_speed_ms, dtype=np.float64), 0.5)
    return SIGMA0_REF_DB + WIND_LAW_DB_PER_DECADE * np.log10(speed / WIND_REF_MS)


def render(
    slick_utm: Polygon,
    centre_utm: np.ndarray,
    crs_epsg: int,
    field: Any,
    to_wgs84: Any,
    t: datetime,
    seed: int,
    damping_db: float = OIL_DAMPING_DB,
) -> SyntheticScene:
    """Render the SC-01 scene around `centre_utm`, with `slick_utm` damped into it.

    `field` is the drift run's own velocity field and `to_wgs84` its inverse
    transformer: the background is sampled from the same wind the solver
    integrated, so the image cannot claim a different sea state than the physics.

    Deterministic given `seed`.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy import ndimage

    n = SCENE_PX
    spacing = PIXEL_SPACING_M
    x0 = float(centre_utm[0]) - SCENE_HALF_SPAN_M
    y1 = float(centre_utm[1]) + SCENE_HALF_SPAN_M
    transform = from_origin(x0, y1, spacing, spacing)

    # --- 5. background from the drift run's own wind field ---------------------
    # Sampled on a coarse mesh and bilinearly upsampled: the synthetic wind field
    # varies over ~100 km, so a 64 x 64 mesh over a 102 km scene resolves it to
    # far better than the speckle floor, and sampling 4.2 M points would cost
    # more than it could possibly reveal.
    mesh = 64
    step = n / mesh
    mx = x0 + (np.arange(mesh) + 0.5) * step * spacing
    my = y1 - (np.arange(mesh) + 0.5) * step * spacing
    grid_x, grid_y = np.meshgrid(mx, my)
    lon, lat = to_wgs84.transform(grid_x.ravel(), grid_y.ravel())
    wind = field.wind10_ms(np.asarray(lon), np.asarray(lat), t)
    wind_speed = np.hypot(wind[:, 0], wind[:, 1]).reshape(mesh, mesh)

    coarse_db = np.asarray(sigma0_for_wind(wind_speed), dtype=np.float64)
    sigma0_db = ndimage.zoom(coarse_db, n / mesh, order=1, mode="nearest")[:n, :n]

    # --- 6. range trend, about the scene centre -------------------------------
    across_km = (np.arange(n) * spacing - SCENE_HALF_SPAN_M) / 1000.0
    sigma0_db = sigma0_db + across_km[np.newaxis, :] * RANGE_TREND_DB_PER_KM

    background_mean_db = float(sigma0_db.mean())

    # --- 1. dark-region geometry: the polygon, and nothing else ---------------
    mask = rasterize(
        [(slick_utm, 1)], out_shape=(n, n), transform=transform, fill=0, dtype="uint8"
    ).astype(np.float64)
    covered_px = int(mask.sum())
    if covered_px == 0:
        raise ValueError(
            "The slick polygon does not overlap the synthetic scene footprint; "
            "the raster would be plain sea and contrast_db would measure nothing."
        )

    # --- 3. one resolution cell of imaging response ---------------------------
    mask = ndimage.gaussian_filter(mask, EDGE_SMOOTH_CELLS, mode="constant")

    # --- 2. damping -----------------------------------------------------------
    sigma0_db = sigma0_db - damping_db * mask

    # --- 4. speckle, in intensity where it belongs ----------------------------
    # Multiplicative Gamma on intensity, then back to dB. Adding Gaussian noise
    # in dB would be the wrong distribution and would make the slick interior and
    # the open sea equally noisy, which is not what multi-look SAR does.
    rng = np.random.default_rng(seed)
    enl = EQUIVALENT_NUMBER_OF_LOOKS
    intensity = 10.0 ** (sigma0_db / 10.0)
    intensity *= rng.gamma(enl, 1.0 / enl, size=(n, n))
    sigma0_db = 10.0 * np.log10(np.maximum(intensity, 1.0e-12))

    method = {
        "badge": BADGE,
        "tier": "§0 Tier 4 — synthetic SAR, elected explicitly for SC-01",
        "not_a_claim": (
            "Scenario imagery for the demo. Not evidence that detection works on "
            "real SAR, and no detection metric may be quoted from it (§16)."
        ),
        "dark_region_geometry": (
            "the slick polygon itself — alpha hull of the real 6 h forward drift "
            "run — rasterised; no shape is drawn here"
        ),
        "slick_pixels": covered_px,
        "damping_db": round(damping_db, 3),
        "damping_rationale": (
            f"midpoint of the {OIL_DAMPING_RANGE_DB[0]}–{OIL_DAMPING_RANGE_DB[1]} dB "
            "range CLAUDE.md states for mineral oil against the surrounding sea; "
            "deliberately not derived from the scorer's own contrast pivot, which "
            "would be circular"
        ),
        "edge_smoothing_cells": EDGE_SMOOTH_CELLS,
        "edge_rationale": (
            "one resolution cell — the oil boundary is sharp, so what is modelled "
            "is the imaging and multilooking response, not the oil"
        ),
        "speckle_distribution": "Gamma on intensity (multi-look SAR intensity)",
        "equivalent_number_of_looks": round(enl, 1),
        "speckle_rationale": (
            f"IW GRDH ~{GRDH_LOOKS} looks at {GRDH_PIXEL_M:.0f} m, resampled to "
            f"{spacing:.0f} m averages {(spacing / GRDH_PIXEL_M) ** 2:.0f} pixels"
        ),
        "background_source": (
            "sampled from the drift run's own wind field, so the image and the "
            "physics cannot disagree about the sea state"
        ),
        "wind_law": (
            f"Sigma0_VV ~ U^{WIND_EXPONENT} pinned at {SIGMA0_REF_DB} dB for "
            f"{WIND_REF_MS} m/s at {INCIDENCE_ANGLE_DEG}° — a first-order power "
            "law, NOT CMOD"
        ),
        "range_trend_db_per_km": round(RANGE_TREND_DB_PER_KM, 4),
        "range_trend_rationale": (
            f"~{SWATH_FALLOFF_DB:.0f} dB across the {SWATH_WIDTH_KM:.0f} km IW swath; "
            "this is why contrast is measured against a local annulus, not a "
            "scene-wide mean"
        ),
        "background_mean_db_before_speckle": round(background_mean_db, 3),
        "seed": int(seed),
        "shape_px": [n, n],
        "pixel_spacing_m": spacing,
        "incidence_angle_deg": INCIDENCE_ANGLE_DEG,
    }

    logger.info(
        "synthetic Sigma0: %d x %d @ %.0f m, background %.2f dB, damping %.2f dB, ENL %.0f",
        n,
        n,
        spacing,
        background_mean_db,
        damping_db,
        enl,
    )
    return SyntheticScene(
        sigma0_db=sigma0_db,
        transform=transform,
        crs_epsg=crs_epsg,
        pixel_spacing_m=spacing,
        incidence_angle_deg=INCIDENCE_ANGLE_DEG,
        method=method,
    )


def write_geotiff(scene: SyntheticScene, path: Path) -> int:
    """Write the raster as int16 hundredths of a dB. Returns the file size."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.rint(scene.sigma0_db * DB_SCALE).astype(np.int16)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=scene.shape[0],
        width=scene.shape[1],
        count=1,
        dtype="int16",
        crs=f"EPSG:{scene.crs_epsg}",
        transform=scene.transform,
        tiled=True,
        compress="deflate",
        predictor=2,
        zlevel=9,
        nodata=int(NODATA_DB * DB_SCALE),
    ) as handle:
        handle.write(data, 1)
        handle.update_tags(
            BADGE=BADGE,
            SCALE_FACTOR=str(1.0 / DB_SCALE),
            UNITS="dB (Sigma0 VV), stored as int16 hundredths",
            PROVENANCE=scene.method["not_a_claim"],
        )

    size = path.stat().st_size
    logger.info("wrote %s (%.1f MB)", path, size / 1.0e6)
    return size
