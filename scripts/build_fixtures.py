"""Capture real engine output into data/fixtures/ (`make fixtures`).

HUMAN-INVOKED ONLY (§13). Per §15 this runs at hour 5, once the drift solver and
the attribution engine can run end-to-end - well enough to emit real output, not
polished.

It writes, per scenario: detections + features, drift frames at 30-min steps,
ranked attributions with channel breakdowns, and the full WS event sequence.

Every number captured here is a genuine engine output. Nothing in this file
fabricates a value - §2.2 applies here above all, because these fixtures are what
the demo falls back to and what the frontend is built against. Where a value
cannot be computed it is emitted as null with a reason in an `unavailable` map,
never as a plausible-looking placeholder.

PROVISIONAL SET - NOT YET FROZEN
--------------------------------
§15 specifies fixtures generated from real engine runs on the real Part III
scenes. Zenodo is down and data/scenes/ and data/ais/ are empty, so this set
runs the real physics and the real fusion over an AUTHORED slick and an AUTHORED
AIS frame. It unblocks §15's purpose - the frontend contract - but it is not the
§9-frozen set and says so in its own manifest. Regenerate and freeze once Tier 1
or Tier 2 data lands.

This is not §0 Tier 4. No synthetic SAR pixels are produced anywhere; there is no
raster at all. The slick is a polygon authored in metric space, and the Tier 4
decision gate is untouched.

How SC-01 is built - a closed loop, not a plant
-----------------------------------------------
    authored release corridor --REAL forward solver 6 h--> observed slick hull
                                                                   |
                                                       REAL backward solver 12 h
                                                                   v
                                                          origin cloud O(x, y, t)
                                                                   |
                                            score_vessels + frame_priors + fuse
                                                                   v
                                                    ranked candidates + verdict

Nothing about the culprit is handed to the attribution engine: it sees the
observed slick and every vessel in the frame on equal terms. The authored release
point and time are recorded in truth.json alongside the recovery error, so the
claim "the backward solver recovered the release point to within X km" is
measured rather than asserted.

The frame is authored once, before the first run, for realism. It is not tuned
afterwards and neither are the §5.4 weights or the §9 decision bands: whatever
verdict the run produces is what ships, quoted with its frame size (§12).
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import transform as shapely_transform

from app.attribution import channels as channels_mod
from app.attribution import fusion as fusion_mod
from app.attribution import priors as priors_mod
from app.config import settings
from app.detection import discriminator as discriminator_mod
from app.detection import explain as explain_mod
from app.detection import features as features_mod
from app.drift import density as density_mod
from app.drift import fields as fields_mod
from app.drift import solver as solver_mod

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_fixtures")

GENERATOR_VERSION = 1

# --------------------------------------------------------------- authored ----
# Everything in this block is a human-authored scenario parameter, not a
# measurement. It is listed here, in one place, so that a reader can see exactly
# what was assumed and what was computed from it.

# Bornholm Basin, open water, inside Danish AIS coverage.
AOI_LON = 15.00
AOI_LAT = 55.20

# Scene acquisition. Sentinel-1 descending passes over the Baltic fall near
# 04:00 UTC, which is also what makes SC-01 a night discharge.
T0 = datetime(2024, 3, 14, 3, 40, tzinfo=UTC)

CULPRIT_MMSI = 220517000
CULPRIT_COG_DEG = 60.0
CULPRIT_SOG_KN = 12.0
CULPRIT_TYPE = "Crude Oil Tanker"

# The discharge runs for an hour and ends 6 h before the scene. The release is
# modelled as the swept corridor along the track over that hour, seeded at the
# moment the discharge completes - which is what discharge-while-underway looks
# like and is exactly the geometry E3 reasons about (L_released = SOG * tau).
RELEASE_DURATION_MIN = 60
RELEASE_AGE_H = 6
RELEASE_HALF_WIDTH_M = 150.0

# Transponder gap straddling the release. 90 min measured; E4 subtracts the
# nominal cadence before scoring it (§5.3).
CULPRIT_GAP_START_MIN = -425
CULPRIT_GAP_END_MIN = -335

AIS_CADENCE_MIN = 15
AIS_SPAN_BEFORE_H = 13
AIS_SPAN_AFTER_H = 1

WINDAGE_VARIANTS = (0.020, 0.033, 0.040)
DEFAULT_WINDAGE = settings.windage_alpha

# Padding around the slick when asking fields.resolve for an ocean state. Wide
# enough that the 12 h cloud never leaves it, and it is what sets the synthetic
# field's mesoscale: ~0.5 deg gives eddy radii in the tens of kilometres, which is
# the right order for the Baltic.
FIELD_BBOX_PAD_DEG = 0.5

# Display raster: the engine grid block-summed by this factor. Exact and
# mass-preserving; E1 ran at full resolution.
DENSITY_COARSEN = 4
DENSITY_MASS_KEPT = 0.99
VARIANT_PARTICLE_SAMPLE = 1000
COORD_DP = 5

SEED_SC01 = 20260314

SIMULATED_AIS_BADGE = "INJECTED — SIMULATED"
AUTHORED_SCENE_BADGE = "SIMULATED SLICK — NO SAR IMAGERY IN THIS BUILD"

# Owned by the modules that produce them, so a reason cannot drift between what
# the engine says and what the fixture records.
NO_PIXELS = features_mod.NO_PIXELS
NO_COASTLINE = features_mod.NO_COASTLINE
NO_MODEL_VERSION = discriminator_mod.NO_MODEL_VERSION
AUTHORED_POLYGON = "the slick polygon is authored for this build; no segmenter ran (§15)"


@dataclass(frozen=True)
class VesselSpec:
    """One authored AIS track. `across_m` / `along_m` are relative to the release
    point, across and along the culprit's heading."""

    mmsi: int
    name: str
    type: str
    length_m: int
    width_m: int
    flag: str
    across_m: float
    along_m: float
    cog_deg: float
    sog_kn: float
    prior_detections: int = 0


# A Bornholm Basin frame: the culprit, four near misses and eighteen vessels going
# about their business. The near misses are the point - a frame where every
# innocent vessel is comfortably far away proves nothing about a method meant to
# run where traffic is dense.
CULPRIT = VesselSpec(
    mmsi=CULPRIT_MMSI,
    name="NORD ASTRAEA",
    type=CULPRIT_TYPE,
    length_m=183,
    width_m=32,
    flag="DK",
    across_m=0.0,
    along_m=0.0,
    cog_deg=CULPRIT_COG_DEG,
    sog_kn=CULPRIT_SOG_KN,
    prior_detections=1,
)

# mmsi, name, AIS type text, length_m, width_m, flag, across_m, along_m, cog_deg, sog_kn
_FRAME_TABLE: tuple[tuple[Any, ...], ...] = (
    # --- near misses: they cross the reconstructed origin cone -----------------
    (219018273, "KATTEGAT TRADER", "Cargo", 142, 22, "DK", 2100.0, -1400.0, 74.0, 11.5),
    (265611000, "SVEA HORIZON", "Cargo", 168, 26, "SE", -3200.0, 2600.0, 208.0, 13.0),
    (211447910, "HANSA BALTICA", "Container Ship", 155, 24, "DE", 4600.0, 900.0, 312.0, 15.5),
    (261019400, "GRYF", "Fishing Vessel", 28, 8, "PL", -5400.0, -3100.0, 41.0, 4.5),
    # --- the rest of the basin -------------------------------------------------
    (219004521, "ESBJERG STAR", "Cargo", 121, 19, "DK", 9200.0, 5400.0, 96.0, 12.0),
    (220389000, "DANA VIKING", "Crude Oil Tanker", 176, 30, "DK", -11800.0, -6200.0, 244.0, 13.5),
    (265520000, "MALMO LINK", "Passenger Ferry", 142, 24, "SE", 14100.0, -9300.0, 18.0, 17.5),
    (211288610, "RUGEN CARRIER", "Bulk Carrier", 189, 31, "DE", -15600.0, 7800.0, 286.0, 10.5),
    (219221000, "NORDSOEN", "Fishing Vessel", 34, 9, "DK", 17900.0, 3200.0, 134.0, 5.0),
    (261084300, "SZCZECIN PIONEER", "Cargo", 137, 21, "PL", -19400.0, -12100.0, 62.0, 11.0),
    (265037000, "GOTLAND SPIRIT", "Passenger Ferry", 155, 25, "SE", 21700.0, 14600.0, 197.0, 19.0),
    (220611000, "SKAGEN PROVIDER", "Cargo", 118, 18, "DK", -23300.0, 4100.0, 351.0, 9.5),
    (211539200, "WARNOW TRADER", "Cargo", 146, 23, "DE", 25800.0, -16400.0, 118.0, 12.5),
    (219170000, "BORNHOLM SUPPLIER", "Cargo", 96, 16, "DK", -27100.0, 9700.0, 271.0, 8.5),
    (265744000, "YSTAD TRAWLER", "Fishing Vessel", 26, 8, "SE", 29400.0, 6300.0, 55.0, 3.5),
    (261230700, "GDYNIA EXPRESS", "Container Ship", 172, 28, "PL", -31900.0, -18200.0, 84.0, 16.0),
    (220455000, "JUTLANDIA", "Crude Oil Tanker", 195, 33, "DK", 33600.0, 11800.0, 226.0, 12.0),
    (211672400, "KIEL MERCHANT", "Bulk Carrier", 181, 30, "DE", -35200.0, -7400.0, 149.0, 10.0),
    (219336000, "HELSINGOR", "Passenger Ferry", 128, 22, "DK", 37800.0, -21600.0, 305.0, 18.5),
    (265892000, "KARLSKRONA NET", "Fishing Vessel", 31, 9, "SE", -39100.0, 16900.0, 7.0, 4.0),
    (261445100, "BALTIC AMBER", "Cargo", 133, 20, "PL", 41500.0, 19400.0, 172.0, 11.5),
    (211804500, "LUEBECK STAR", "Cargo", 159, 25, "DE", -43700.0, -13500.0, 29.0, 14.0),
)

OTHER_VESSELS: tuple[VesselSpec, ...] = tuple(VesselSpec(*row) for row in _FRAME_TABLE)

# --- SC-02: a low-wind look-alike ------------------------------------------
SC02_WIND_MS = 2.1
SC02_CENTRE_LON = 14.62
SC02_CENTRE_LAT = 55.48
SC02_RADIUS_M = 3100.0
SC02_T0 = datetime(2024, 3, 14, 3, 42, tzinfo=UTC)
SEED_SC02 = 20260315

# --- SC-03: clean sea -------------------------------------------------------
SC03_CENTRE_LON = 15.41
SC03_CENTRE_LAT = 54.95
SC03_WIND_MS = 7.4
SC03_T0 = datetime(2024, 3, 14, 3, 44, tzinfo=UTC)

SCENE_HALF_SPAN_M = 51_200.0  # 2048 px x 50 m, the Part III scene footprint size


# ------------------------------------------------------------- geo helpers ----


def _to_utm(polygon: Polygon, frame: solver_mod.Frame) -> Polygon:
    return shapely_transform(lambda x, y: frame.to_utm.transform(x, y), polygon)


def _to_wgs84(polygon: Polygon, frame: solver_mod.Frame) -> Polygon:
    return shapely_transform(lambda x, y: frame.to_wgs84.transform(x, y), polygon)


def _round_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    def walk(node: Any) -> Any:
        if isinstance(node, list | tuple):
            if node and isinstance(node[0], int | float):
                return [round(float(v), COORD_DP) for v in node]
            return [walk(item) for item in node]
        return node

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def _geojson(polygon: Polygon, frame: solver_mod.Frame | None = None) -> dict[str, Any]:
    """GeoJSON in EPSG:4326 (§8). `frame` given means the polygon is in its UTM."""
    wgs84 = polygon if frame is None else _to_wgs84(polygon, frame)
    return _round_geometry(mapping(wgs84))


def _bearing_deg(dx_east_m: float, dy_north_m: float) -> float:
    """Compass bearing of a metric vector, folded to [0, 180).

    A slick axis is undirected, and E2 folds at 90 degrees anyway; keeping the
    convention identical to COG (0 = north, clockwise) is what makes the two
    commensurable.
    """
    return float(math.degrees(math.atan2(dx_east_m, dy_north_m)) % 180.0)


def _wind_from_field(
    field: fields_mod.VelocityField, lon: float, lat: float, t: datetime
) -> tuple[float, float]:
    """Wind speed and meteorological direction at a point, read off the field itself.

    Reported rather than authored: the scene's wind is the wind the drift run
    actually integrated, so the §5.2 gate and the physics cannot disagree.
    Direction follows the meteorological convention — the bearing the wind blows
    FROM, which is what an AIS-era operator reads.
    """
    vector = field.wind10_ms(np.array([lon]), np.array([lat]), t)[0]
    east, north = float(vector[0]), float(vector[1])
    speed_ms = math.hypot(east, north)
    direction_deg = (math.degrees(math.atan2(east, north)) + 180.0) % 360.0
    return round(speed_ms, 2), round(direction_deg, 1)


def _direction(cog_deg: float) -> np.ndarray:
    heading = math.radians(cog_deg)
    return np.array([math.sin(heading), math.cos(heading)])


def _perpendicular(cog_deg: float) -> np.ndarray:
    heading = math.radians(cog_deg)
    return np.array([math.cos(heading), -math.sin(heading)])


def _hull_geometry(polygon_utm: Polygon) -> dict[str, float]:
    """Geometry features of the observed slick, in a local UTM frame (§8).

    Thin rounding wrapper over `detection.features.geometry_features`, which is
    the one implementation. Two copies of a feature definition means one of them
    is wrong and nobody knows which — the same reason §0 keeps the data split in
    exactly one place.
    """
    return {
        name: round(value, 2 if name == "orientation_deg" else 4)
        for name, value in features_mod.geometry_features(polygon_utm).items()
    }


# ----------------------------------------------------------- AIS authoring ----


def _ais_times(t0: datetime) -> list[datetime]:
    start = t0 - timedelta(hours=AIS_SPAN_BEFORE_H)
    n = int((AIS_SPAN_BEFORE_H + AIS_SPAN_AFTER_H) * 60 / AIS_CADENCE_MIN) + 1
    return [start + timedelta(minutes=AIS_CADENCE_MIN * i) for i in range(n)]


def _track_positions_m(
    spec: VesselSpec,
    times: list[datetime],
    reference_m: np.ndarray,
    reference_time: datetime,
    reference_cog_deg: float,
) -> np.ndarray:
    """A constant-course, constant-speed track through an offset reference point.

    Offsets are taken across and along the culprit's heading so that the frame
    can be authored in plain terms - "2.1 km off the discharge track" - and stay
    meaningful whatever the release geometry.
    """
    origin = (
        reference_m
        + _perpendicular(reference_cog_deg) * spec.across_m
        + _direction(reference_cog_deg) * spec.along_m
    )
    heading = _direction(spec.cog_deg)
    speed_ms = spec.sog_kn * 1852.0 / 3600.0
    return np.array(
        [origin + heading * speed_ms * (t - reference_time).total_seconds() for t in times]
    )


def _build_frame_tracks(
    specs: list[VesselSpec],
    frame: solver_mod.Frame,
    t0: datetime,
    reference_m: np.ndarray,
    reference_time: datetime,
    reference_cog_deg: float,
    gaps: dict[int, tuple[datetime, datetime]],
) -> tuple[list[channels_mod.VesselTrack], dict[int, list[dict[str, Any]]]]:
    """VesselTracks in the run's metric frame, plus their WGS84 serialisation.

    `channels.VesselTrack` is fed metric positions directly: it is the only
    implementation of track interpolation and gap measurement in the codebase and
    is not reimplemented here.
    """
    all_times = _ais_times(t0)
    tracks: list[channels_mod.VesselTrack] = []
    serialised: dict[int, list[dict[str, Any]]] = {}

    for spec in specs:
        gap = gaps.get(spec.mmsi)
        times = [t for t in all_times if gap is None or not gap[0] < t < gap[1]]
        positions_m = _track_positions_m(
            spec, times, reference_m, reference_time, reference_cog_deg
        )
        tracks.append(
            channels_mod.VesselTrack(
                mmsi=spec.mmsi,
                times=times,
                positions_m=positions_m,
                sog_kn=np.full(len(times), spec.sog_kn),
                cog_deg=np.full(len(times), spec.cog_deg),
            )
        )
        lon, lat = frame.inverse(positions_m)
        serialised[spec.mmsi] = [
            {
                "ts": t.isoformat(),
                "lon": round(float(x), COORD_DP),
                "lat": round(float(y), COORD_DP),
                "sog_kn": spec.sog_kn,
                "cog_deg": spec.cog_deg,
                "is_injected": True,
            }
            for t, x, y in zip(times, lon, lat, strict=True)
        ]

    return tracks, serialised


def _serialise_ais(
    specs: list[VesselSpec],
    tracks: list[channels_mod.VesselTrack],
    serialised: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_mmsi = {track.mmsi: track for track in tracks}
    vessels = []
    for spec in specs:
        track = by_mmsi[spec.mmsi]
        positions = serialised[spec.mmsi]
        gaps_min = [
            (b - a).total_seconds() / 60.0
            for a, b in zip(track.times, track.times[1:], strict=False)
        ]
        vessels.append(
            {
                "mmsi": spec.mmsi,
                "name": spec.name,
                "type": spec.type,
                "canonical_type": priors_mod.canonical_type(spec.type),
                "length_m": spec.length_m,
                "width_m": spec.width_m,
                "flag": spec.flag,
                "prior_detections": spec.prior_detections,
                "is_injected": True,
                "badge": SIMULATED_AIS_BADGE,
                "t_start": track.t_start.isoformat(),
                "t_end": track.t_end.isoformat(),
                "n_positions": len(positions),
                "gap_count": sum(1 for g in gaps_min if g > AIS_CADENCE_MIN),
                "max_gap_min": round(max(gaps_min), 1) if gaps_min else 0.0,
                "positions": positions,
            }
        )

    return {
        "provenance": "authored — no real Danish AIS on disk in this build",
        "note": (
            "Every track in this frame is injected, not only the culprit's. When real "
            "Danish AIS lands, only the culprit stays injected and this note goes away."
        ),
        "badge": SIMULATED_AIS_BADGE,
        "cadence_min": AIS_CADENCE_MIN,
        "cadence_note": (
            "Sample cadence equals settings.e4_nominal_cadence_min, so a clean "
            "transmitter scores exactly s4 = 1.0 (§5.3)."
        ),
        "n_vessels": len(vessels),
        "vessels": vessels,
    }


# --------------------------------------------------------- drift serialising ----


def _coarsen(probability: np.ndarray, factor: int) -> np.ndarray:
    """Block-sum the engine raster. Exact and mass-preserving, not resampled."""
    ny, nx = probability.shape
    pad_y = (-ny) % factor
    pad_x = (-nx) % factor
    padded = np.pad(probability, ((0, pad_y), (0, pad_x)), mode="constant")
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    return blocks.sum(axis=(1, 3))


def _sparse_density(coarse: np.ndarray) -> tuple[list[list[float]], float]:
    """Cells carrying the top DENSITY_MASS_KEPT of this snapshot's mass.

    Returns the cells and the mass actually retained, so the §5.1 run-wide
    normalisation stays checkable from the fixture alone.
    """
    flat = coarse.ravel()
    order = np.argsort(flat)[::-1]
    ordered = flat[order]
    total = float(ordered.sum())
    if total <= 0.0:
        return [], 0.0

    cumulative = np.cumsum(ordered)
    cutoff = int(np.searchsorted(cumulative, DENSITY_MASS_KEPT * total) + 1)
    keep = order[:cutoff]
    iy, ix = np.unravel_index(keep, coarse.shape)
    retained = float(flat[keep].sum())
    return (
        [[int(x), int(y), float(f"{flat[k]:.6e}")] for x, y, k in zip(ix, iy, keep, strict=True)],
        retained,
    )


def _serialise_drift(
    run: solver_mod.DriftRun,
    run_density: density_mod.RunDensity,
    max_particles: int | None,
) -> dict[str, Any]:
    grid = run_density.grid
    coarse_cell_m = grid.cell_size_m * DENSITY_COARSEN

    if max_particles is not None and max_particles < run.n_particles:
        # Deterministic stride, not a random draw: the subsample must be identical
        # on every machine for the offline demo to replay the same picture (§2.1).
        stride = math.ceil(run.n_particles / max_particles)
        selection: slice | None = slice(None, None, stride)
    else:
        selection = None

    frames = []
    retained_total = 0.0
    for index, snapshot in enumerate(run_density.snapshots):
        lonlat = run.positions_wgs84(index)
        if selection is not None:
            lonlat = lonlat[selection]
        coarse = _coarsen(snapshot.probability, DENSITY_COARSEN)
        cells, retained = _sparse_density(coarse)
        retained_total += retained

        frames.append(
            {
                "t_offset_min": snapshot.t_offset_min,
                "t": (run.t0 + timedelta(minutes=snapshot.t_offset_min)).isoformat(),
                "stretch_factor": round(snapshot.stretch_factor, 6),
                "stretch_factor_raw": round(snapshot.stretch_factor_raw, 6),
                "n_particles": int(lonlat.shape[0]),
                "particles": [
                    [round(float(x), COORD_DP), round(float(y), COORD_DP)] for x, y in lonlat
                ],
                "hull": _geojson(snapshot.hull, run.frame),
                "hull_area_km2": round(snapshot.hull.area / 1.0e6, 4),
                "density_cells": cells,
            }
        )

    payload: dict[str, Any] = {
        "mode": run.mode,
        "t0": run.t0.isoformat(),
        "horizon_h": run.horizon_h,
        "n_particles": run.n_particles,
        "windage": run.windage,
        "k_h_m2s": run.k_h,
        "theta_dev_deg": run.theta_dev_deg,
        "dt_s": settings.drift_dt_s,
        "snapshot_interval_min": settings.drift_snapshot_interval_min,
        "field_source": run.field_source,
        "seed": run.seed,
        "n_frames": len(frames),
        "density_grid": {
            "crs": f"EPSG:{run.frame.crs.to_epsg()}",
            "x0_m": round(grid.x0_m, 3),
            "y0_m": round(grid.y0_m, 3),
            "cell_size_m": coarse_cell_m,
            "engine_cell_size_m": grid.cell_size_m,
            "nx": int(math.ceil(grid.nx / DENSITY_COARSEN)),
            "ny": int(math.ceil(grid.ny / DENSITY_COARSEN)),
            "coarsen_factor": DENSITY_COARSEN,
            "note": (
                "Display raster: the engine grid block-summed, which is exact and "
                "mass-preserving. E1 integrated the full-resolution field."
            ),
            "mass_kept_fraction": DENSITY_MASS_KEPT,
            "mass_retained_run_total": round(retained_total, 6),
        },
        "frames": frames,
    }
    if selection is not None:
        payload["particles_subsampled_from"] = run.n_particles
        payload["particles_subsample_stride"] = selection.step
    return payload


# --------------------------------------------------- attribution serialising ----


def _serialise_channels(scores: channels_mod.ChannelScores) -> dict[str, Any]:
    unavailable = dict(scores.unavailable)
    if scores.t_star is None:
        reason = unavailable.get("s2", channels_mod.NO_ORIGIN_OVERLAP)
        # t* is E1's output and everything else anchors to it, so when it is
        # absent the absence is the finding, not a gap in the record.
        unavailable.setdefault("t_star", reason)
        unavailable.setdefault("stretch_factor", reason)
    return {
        "mmsi": scores.mmsi,
        "s1": round(scores.s1, 6),
        "s2": None if scores.s2 is None else round(scores.s2, 6),
        "s3": None if scores.s3 is None else round(scores.s3, 6),
        "s4": None if scores.s4 is None else round(scores.s4, 6),
        "mass": round(scores.mass, 9),
        "t_star": None if scores.t_star is None else scores.t_star.isoformat(),
        "stretch_factor": (
            None if scores.stretch_factor is None else round(scores.stretch_factor, 6)
        ),
        "unavailable": unavailable,
    }


def _serialise_attribution(
    result: fusion_mod.FrameResult,
    names: dict[int, str],
    n_vessels_in_frame: int,
    windage: float,
) -> dict[str, Any]:
    culprit = result.culprit
    return {
        "verdict": result.verdict,
        "issued": True,
        "culprit_mmsi": None if culprit is None else culprit.mmsi,
        "windage": windage,
        "n_vessels_in_frame": n_vessels_in_frame,
        "n_ranked": len(result.candidates),
        "frame_size_note": (
            "The E1 term is a ratio against what an average vessel in THIS frame "
            "scores, so achievable LR scales with log N. Quote this N with the LR (§12)."
        ),
        "background": {
            "s1_bg": round(result.background[0], 6),
            "s2_bg": round(result.background[1], 6),
            "s3_bg": round(result.background[2], 6),
        },
        "none_of_the_above_posterior": round(result.none_posterior, 6),
        "decision_bands": {"MODERATE": "LR >= 10", "STRONG": "LR >= 100"},
        "candidates": [
            {
                "rank": candidate.rank,
                "mmsi": candidate.mmsi,
                "name": names.get(candidate.mmsi, ""),
                "log_lr": round(candidate.log_lr, 6),
                "lr": round(candidate.lr, 4),
                "posterior": round(candidate.posterior, 6),
                "verdict": candidate.verdict,
                "prior": round(candidate.prior, 6),
                "terms_nats": {
                    key: None if value is None else round(value, 6)
                    for key, value in candidate.terms.items()
                },
                "channels": _serialise_channels(candidate.channels),
            }
            for candidate in result.candidates
        ],
        "unrankable": [
            {
                "mmsi": entry.mmsi,
                "name": names.get(entry.mmsi, ""),
                "reason": entry.reason,
                "verdict": entry.verdict,
                "channels": _serialise_channels(entry.channels),
            }
            for entry in result.unrankable
        ],
    }


# --------------------------------------------------------- recovery measure ----


def _weighted_centroid_m(
    probability: np.ndarray, grid: density_mod.DensityGrid
) -> tuple[float, float]:
    x, y = grid.cell_centres()
    total = float(probability.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    return (
        float(probability.sum(axis=0) @ x / total),
        float(probability.sum(axis=1) @ y / total),
    )


def _recovery(
    run: solver_mod.DriftRun,
    run_density: density_mod.RunDensity,
    release_centroid_m: np.ndarray,
    release_time: datetime,
    t_star: datetime | None,
    culprit_channels: channels_mod.ChannelScores | None,
    observed_major_axis_km: float,
) -> dict[str, Any]:
    """How close the backward run got to the authored release, in km and minutes.

    Measured in the run's UTM frame, never in degrees (§8). t* is quantised to the
    30-min snapshot interval by construction, so its error is too.

    The released-length block is what makes E3 checkable: the authored discharge
    has a known length (SOG x duration), so `major_axis / stretch_factor` - the
    §5.3 line itself - can be compared against it instead of taken on trust. The
    true stretch is recorded next to the one the solver reported, which is the
    only way to see whether §5.1's floor at 1.0 helped or hurt on this run.
    """
    target_offset = round((release_time - run.t0).total_seconds() / 60.0)
    index = min(
        range(len(run_density.snapshots)),
        key=lambda i: abs(run_density.snapshots[i].t_offset_min - target_offset),
    )
    snapshot = run_density.snapshots[index]
    grid = run_density.grid

    cx, cy = _weighted_centroid_m(snapshot.probability, grid)
    iy, ix = np.unravel_index(int(np.argmax(snapshot.probability)), snapshot.probability.shape)
    xs, ys = grid.cell_centres()
    peak = np.array([xs[ix], ys[iy]])

    return {
        "snapshot_t_offset_min": snapshot.t_offset_min,
        "snapshot_t": (run.t0 + timedelta(minutes=snapshot.t_offset_min)).isoformat(),
        "origin_error_km": round(
            float(np.hypot(cx - release_centroid_m[0], cy - release_centroid_m[1])) / 1000.0, 3
        ),
        "origin_peak_error_km": round(
            float(np.hypot(*(peak - release_centroid_m))) / 1000.0, 3
        ),
        "release_inside_hull": bool(
            snapshot.hull.contains(Point(release_centroid_m[0], release_centroid_m[1]))
        ),
        "hull_area_km2": round(snapshot.hull.area / 1.0e6, 4),
        "t_star": None if t_star is None else t_star.isoformat(),
        "t_star_error_min": (
            None if t_star is None else round((t_star - release_time).total_seconds() / 60.0, 1)
        ),
        "t_star_resolution_min": settings.drift_snapshot_interval_min,
        **_released_length(culprit_channels, observed_major_axis_km),
        "note": (
            "Origin error is the distance from the authored release centroid to the "
            "backward run's density-weighted centroid at the snapshot nearest the "
            "release time. t* is quantised to the snapshot interval."
        ),
    }


# ------------------------------------------------------------------ scenes ----


def _released_length(
    culprit_channels: channels_mod.ChannelScores | None,
    observed_major_axis_km: float,
) -> dict[str, Any]:
    """E3's released-patch reconstruction against the authored discharge."""
    authored_km = CULPRIT_SOG_KN * channels_mod.KNOTS_TO_KMH * (RELEASE_DURATION_MIN / 60.0)
    true_stretch = observed_major_axis_km / authored_km if authored_km > 0.0 else None

    block: dict[str, Any] = {
        "released_length_km_authored": round(authored_km, 3),
        "observed_major_axis_km": round(observed_major_axis_km, 3),
        "true_stretch_factor": None if true_stretch is None else round(true_stretch, 4),
    }
    if culprit_channels is None or culprit_channels.stretch_factor is None:
        block["released_length_km_recovered"] = None
        block["released_length_error_km"] = None
        block["stretch_factor_at_t_star"] = None
        block["unavailable"] = {
            "released_length_km_recovered": "the culprit has no t*, so E3 had no stretch factor"
        }
        return block

    stretch = culprit_channels.stretch_factor
    recovered_km = observed_major_axis_km / stretch
    block["released_length_km_recovered"] = round(recovered_km, 3)
    block["released_length_error_km"] = round(recovered_km - authored_km, 3)
    block["stretch_factor_at_t_star"] = round(stretch, 4)
    block["s3_at_t_star"] = (
        None if culprit_channels.s3 is None else round(culprit_channels.s3, 6)
    )
    return block


def _scene(
    code: str,
    name: str,
    t0: datetime,
    centre_lon: float,
    centre_lat: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    frame: solver_mod.Frame,
) -> dict[str, Any]:
    centre_m = frame.forward(np.array([centre_lon]), np.array([centre_lat]))[0]
    box = shapely.box(
        centre_m[0] - SCENE_HALF_SPAN_M,
        centre_m[1] - SCENE_HALF_SPAN_M,
        centre_m[0] + SCENE_HALF_SPAN_M,
        centre_m[1] + SCENE_HALF_SPAN_M,
    )
    return {
        "id": code,
        "product_id": f"VARUNA_FIXTURE_{code.replace('-', '')}",
        "name": name,
        "sensor": "Sentinel-1 (footprint authored; no raster in this build)",
        "acq_time": t0.isoformat(),
        "footprint": _geojson(box, frame),
        "centre": {"lon": centre_lon, "lat": centre_lat},
        "incidence_angle_deg": None,
        "wind_speed_ms": wind_speed_ms,
        "wind_dir_deg": wind_dir_deg,
        "pixel_spacing_m": None,
        "source": "authored",
        "raster_path": None,
        "badges": [AUTHORED_SCENE_BADGE],
        "unavailable": {
            "raster_path": "no SAR scene on disk; Zenodo Part III unavailable at build time",
            "incidence_angle_deg": NO_PIXELS,
            "pixel_spacing_m": NO_PIXELS,
        },
    }


def _detection(
    detection_id: str,
    scene_id: str,
    polygon_utm: Polygon,
    frame: solver_mod.Frame,
    wind_speed_ms: float,
    n_ships_within_20km: int | None,
) -> dict[str, Any]:
    """One detection block: real features, a real classification, honest gaps.

    The polygon is authored — no segmenter ran (§15) — but everything computed
    FROM it is engine output. `features.extract` produces the §5.2 vector and
    marks what it could not measure, and `discriminator.score` classifies it.
    With no trained model on disk that classification comes from the rule-based
    §5.2 physics scorer, and the block says so in `method` and `note` rather
    than letting a reader assume a model ran.
    """
    geometry = _hull_geometry(polygon_utm)
    extracted = features_mod.extract(
        polygon_utm,
        sigma0_db=None,
        wind_speed_ms=wind_speed_ms,
        vessel_positions_m=None,
    )
    # The generator knows the AIS frame; features.extract does not receive it.
    values = dict(extracted.values)
    values.update(geometry)
    unavailable = dict(extracted.unavailable)
    if n_ships_within_20km is not None:
        values["n_ships_within_20km"] = n_ships_within_20km
        unavailable.pop("n_ships_within_20km", None)

    extracted = features_mod.ExtractedFeatures(values=values, unavailable=unavailable)
    result = discriminator_mod.score(extracted)
    gate_violated, gate_reason = discriminator_mod.wind_gate(extracted)
    payload = explain_mod.to_ui_payload(extracted, result)

    return {
        "id": detection_id,
        "scene_id": scene_id,
        "geom": _geojson(polygon_utm, frame),
        "features": {
            name: (round(value, 4) if isinstance(value, float) else value)
            for name, value in values.items()
        },
        "feature_provenance": {name: AUTHORED_POLYGON for name in geometry},
        "p_oil": round(result.p_oil, 4) if result.p_oil is not None else None,
        "class": result.p_class,
        "method": result.method,
        "note": result.note,
        "shap_factors": payload["factors"],
        "factor_basis": result.basis,
        "base_p_oil": result.base_p_oil,
        "evidence_fraction": round(result.evidence_fraction, 4),
        "wind_gate_violated": gate_violated,
        "wind_gate_reason": gate_reason,
        "wind_gate_range_ms": [settings.wind_gate_min_ms, settings.wind_gate_max_ms],
        "model_version": None,
        "detector": result.method,
        "unavailable": dict(result.unavailable),
        "provenance": AUTHORED_POLYGON,
    }


def _count_ships_within(
    tracks: list[channels_mod.VesselTrack], centroid_m: np.ndarray, radius_m: float
) -> int:
    return sum(
        1
        for track in tracks
        if float(np.min(np.hypot(*(track.positions_m - centroid_m).T))) <= radius_m
    )


# ------------------------------------------------------------- WS sequence ----


def _t_label(t_offset_min: int) -> str:
    sign = "−" if t_offset_min < 0 else "+"
    minutes = abs(int(t_offset_min))
    return f"T{sign}{minutes // 60}h {minutes % 60:02d}m"


def _event(
    stage: str, progress: float, message: str, playback_ms: int, **payload: Any
) -> dict[str, Any]:
    return {
        "stage": stage,
        "progress": round(progress, 3),
        "message": message,
        "playback_offset_ms": playback_ms,
        "payload": payload,
    }


# ------------------------------------------------------------------ writer ----


def _write_json(path: Path, payload: Any, compact: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=2, ensure_ascii=False)
    )
    path.write_text(text + "\n", encoding="utf-8")
    logger.info("wrote %s (%.1f KB)", path, len(text) / 1024.0)
    return len(text)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


# --------------------------------------------------------------- scenarios ----


def build_sc01(root: Path) -> dict[str, Any]:
    """Baltic Night Discharge — the closed loop (§12 SC-01)."""
    code = "SC-01"
    directory = root / code
    authoring = solver_mod.Frame.for_point(AOI_LON, AOI_LAT)

    release_end_time = T0 - timedelta(hours=RELEASE_AGE_H)
    release_start_time = release_end_time - timedelta(minutes=RELEASE_DURATION_MIN)

    release_end_m = authoring.forward(np.array([AOI_LON]), np.array([AOI_LAT]))[0]
    speed_ms = CULPRIT_SOG_KN * 1852.0 / 3600.0
    release_start_m = release_end_m - _direction(CULPRIT_COG_DEG) * speed_ms * (
        RELEASE_DURATION_MIN * 60
    )
    corridor_utm = LineString([release_start_m, release_end_m]).buffer(
        RELEASE_HALF_WIDTH_M, cap_style="flat"
    )
    corridor_wgs84 = _to_wgs84(corridor_utm, authoring)

    field = fields_mod.resolve(
        corridor_wgs84.buffer(FIELD_BBOX_PAD_DEG).bounds,
        release_end_time,
        scenario_code=code,
        seed=SEED_SC01,
    )

    # --- forward leg: the release becomes the observed slick -------------------
    forward_run = solver_mod.run(
        corridor_wgs84,
        "forward",
        field,
        t0=release_end_time,
        horizon_h=RELEASE_AGE_H,
        seed=SEED_SC01,
    )
    observed_utm = density_mod.alpha_hull(forward_run.states[-1].positions)
    observed_wgs84 = _to_wgs84(observed_utm, forward_run.frame)

    # --- backward leg: what the pipeline actually sees --------------------------
    runs: dict[float, solver_mod.DriftRun] = {}
    densities: dict[float, density_mod.RunDensity] = {}
    for windage in WINDAGE_VARIANTS:
        run = solver_mod.run(
            observed_wgs84, "backward", field, t0=T0, windage=windage, seed=SEED_SC01
        )
        runs[windage] = run
        densities[windage] = density_mod.compute_run_density(run.states)

    default_run = runs[DEFAULT_WINDAGE]
    default_density = densities[DEFAULT_WINDAGE]
    geometry = _hull_geometry(_to_utm(observed_wgs84, default_run.frame))

    # --- the frame ---------------------------------------------------------------
    specs = [CULPRIT, *OTHER_VESSELS]
    release_end_lon, release_end_lat = authoring.to_wgs84.transform(
        float(release_end_m[0]), float(release_end_m[1])
    )
    release_end_in_run = default_run.frame.forward(
        np.array([release_end_lon]), np.array([release_end_lat])
    )[0]
    gaps = {
        CULPRIT_MMSI: (
            T0 + timedelta(minutes=CULPRIT_GAP_START_MIN),
            T0 + timedelta(minutes=CULPRIT_GAP_END_MIN),
        )
    }
    tracks, serialised_tracks = _build_frame_tracks(
        specs, default_run.frame, T0, release_end_in_run, release_end_time, CULPRIT_COG_DEG, gaps
    )
    vessel_meta = {spec.mmsi: (spec.type, spec.prior_detections) for spec in specs}
    names = {spec.mmsi: spec.name for spec in specs}
    frame_priors = priors_mod.frame_priors(vessel_meta)

    # --- score and fuse, once per windage variant --------------------------------
    fuse_started = datetime.now(tz=UTC)
    results: dict[float, fusion_mod.FrameResult] = {}
    for windage in WINDAGE_VARIANTS:
        scores = channels_mod.score_vessels(
            tracks,
            densities[windage],
            T0,
            slick_orientation_deg=geometry["orientation_deg"],
            major_axis_km=geometry["major_axis_km"],
        )
        results[windage] = fusion_mod.fuse(scores, frame_priors)
    fuse_ms = int((datetime.now(tz=UTC) - fuse_started).total_seconds() * 1000)

    default_result = results[DEFAULT_WINDAGE]

    slick_centroid_m = np.array(
        _to_utm(observed_wgs84, default_run.frame).centroid.coords[0]
    )
    release_centroid_run_m = np.array(
        _to_utm(corridor_wgs84, default_run.frame).centroid.coords[0]
    )

    # --- write ------------------------------------------------------------------
    wind_speed_ms, wind_dir_deg = _wind_from_field(field, AOI_LON, AOI_LAT, T0)
    scene = _scene(
        code,
        "Baltic Night Discharge",
        T0,
        AOI_LON,
        AOI_LAT,
        wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
        frame=default_run.frame,
    )
    detection = _detection(
        f"{code}-D1",
        code,
        _to_utm(observed_wgs84, default_run.frame),
        default_run.frame,
        wind_speed_ms,
        _count_ships_within(tracks, slick_centroid_m, 20_000.0),
    )
    detection["provenance"] = (
        "Alpha hull of a real 6 h forward drift run seeded on the authored release "
        "corridor. The attribution engine sees only this polygon. Features and "
        "classification are engine output; the polygon is not."
    )

    _write_json(directory / "scene.json", scene)
    _write_json(directory / "detection.json", detection)
    _write_json(directory / "ais.json", _serialise_ais(specs, tracks, serialised_tracks))

    for windage in WINDAGE_VARIANTS:
        is_default = windage == DEFAULT_WINDAGE
        suffix = "" if is_default else f".windage-{windage:.3f}"
        _write_json(
            directory / f"drift{suffix}.json",
            _serialise_drift(
                runs[windage],
                densities[windage],
                None if is_default else VARIANT_PARTICLE_SAMPLE,
            ),
            compact=True,
        )
        _write_json(
            directory / f"attribution{suffix}.json",
            _serialise_attribution(results[windage], names, len(specs), windage),
        )

    culprit_channels = next(
        (
            entry.channels
            for entry in (*default_result.candidates, *default_result.unrankable)
            if entry.mmsi == CULPRIT_MMSI
        ),
        None,
    )
    culprit_t_star = None if culprit_channels is None else culprit_channels.t_star
    recovery = _recovery(
        default_run,
        default_density,
        release_centroid_run_m,
        release_end_time,
        culprit_t_star,
        culprit_channels,
        geometry["major_axis_km"],
    )
    release_lon, release_lat = authoring.to_wgs84.transform(
        *_to_utm(corridor_wgs84, authoring).centroid.coords[0]
    )
    _write_json(
        directory / "truth.json",
        {
            "note": (
                "Authored ground truth for the closed-loop scenario, recorded so the "
                "recovery claim is measured rather than asserted. None of it is "
                "visible to the attribution engine."
            ),
            "authored": {
                "culprit_mmsi": CULPRIT_MMSI,
                "culprit_name": CULPRIT.name,
                "culprit_type": CULPRIT_TYPE,
                "release_lon": round(float(release_lon), COORD_DP),
                "release_lat": round(float(release_lat), COORD_DP),
                "release_start": release_start_time.isoformat(),
                "release_end": release_end_time.isoformat(),
                "release_duration_min": RELEASE_DURATION_MIN,
                "release_corridor": _geojson(corridor_wgs84),
                "sog_kn": CULPRIT_SOG_KN,
                "cog_deg": CULPRIT_COG_DEG,
                "transponder_gap_min": CULPRIT_GAP_END_MIN - CULPRIT_GAP_START_MIN,
            },
            "forward_leg": {
                "mode": "forward",
                "horizon_h": RELEASE_AGE_H,
                "n_particles": forward_run.n_particles,
                "field_source": forward_run.field_source,
                "observed_slick": _geojson(observed_wgs84),
                "observed_geometry": geometry,
            },
            "recovered": recovery,
            "identified_culprit_mmsi": (
                None if default_result.culprit is None else default_result.culprit.mmsi
            ),
            "identified_correctly": (
                default_result.culprit is not None
                and default_result.culprit.mmsi == CULPRIT_MMSI
            ),
            "verdict": default_result.verdict,
        },
    )

    events = _ws_sc01(
        scene, detection, geometry, default_run, default_density, default_result, names, len(specs)
    )
    _write_json(
        directory / "ws.json",
        {
            "scenario": code,
            "timings_ms": {
                "forward_drift": int(forward_run.elapsed_s * 1000),
                "backward_drift": int(default_run.elapsed_s * 1000),
                "attribution_all_variants": fuse_ms,
            },
            "timings_note": (
                "Measured engine time on the fixture-build machine. Each event's "
                "playback_offset_ms is authored demo pacing (§12) and is not a timing."
            ),
            "total_playback_ms": events[-1]["playback_offset_ms"],
            "events": events,
        },
    )

    return {
        "code": code,
        "name": "Baltic Night Discharge",
        "verdict": default_result.verdict,
        "culprit_mmsi": (
            None if default_result.culprit is None else default_result.culprit.mmsi
        ),
        "lr": (None if default_result.culprit is None else round(default_result.culprit.lr, 2)),
        "n_vessels_in_frame": len(specs),
        "n_ranked": len(default_result.candidates),
        "origin_error_km": recovery["origin_error_km"],
        "t_star_error_min": recovery["t_star_error_min"],
        "artefacts": sorted(p.stem for p in directory.glob("*.json")),
    }


def _ws_sc01(
    scene: dict[str, Any],
    detection: dict[str, Any],
    geometry: dict[str, float],
    run: solver_mod.DriftRun,
    run_density: density_mod.RunDensity,
    result: fusion_mod.FrameResult,
    names: dict[int, str],
    n_vessels: int,
) -> list[dict[str, Any]]:
    """§12 budgets SC-01 at 110 s. Messages carry numbers this run produced."""
    events = [
        _event(
            "SEGMENTING",
            0.05,
            f"Loading scene {scene['id']} — Bornholm Basin, {T0:%Y-%m-%d %H:%MZ}",
            0,
            scene_id=scene["id"],
            badges=scene["badges"],
        ),
        _event(
            "SEGMENTING",
            0.18,
            f"1 dark formation delineated — {geometry['area_km2']:.1f} km², major axis "
            f"{geometry['major_axis_km']:.1f} km, bearing {geometry['orientation_deg']:.0f}°",
            6_000,
            detection_id=detection["id"],
            provisional=True,
            note="Slick polygon is authored for this build; no segmenter ran (§15).",
        ),
        _event(
            "DISCRIMINATING",
            0.28,
            f"Classified {detection['class']} — P(oil) {detection['p_oil']:.2f} "
            f"(rule-based §5.2 physics; no trained model in this build)",
            16_000,
            p_oil=detection["p_oil"],
            p_class=detection["class"],
            method=detection["method"],
            note=detection["note"],
            factors=detection["shap_factors"],
            wind_gate_violated=detection["wind_gate_violated"],
            unavailable=detection["unavailable"],
        ),
    ]

    last = len(run_density.snapshots) - 1
    rewind_indices = [min(i, last) for i in (4, 8, 12, 16, 20, 24)]
    for step, index in enumerate(rewind_indices):
        snapshot = run_density.snapshots[index]
        events.append(
            _event(
                "REWINDING",
                0.35 + 0.35 * (step + 1) / len(rewind_indices),
                f"Rewinding ocean state to {_t_label(snapshot.t_offset_min)}",
                24_000 + step * 7_000,
                t_offset_min=snapshot.t_offset_min,
                hull_area_km2=round(snapshot.hull.area / 1.0e6, 2),
                stretch_factor=round(snapshot.stretch_factor, 3),
                n_particles=run.n_particles,
                field_source=run.field_source,
            )
        )

    events.append(
        _event(
            "REWINDING",
            0.74,
            f"Origin cloud reconstructed over {run.horizon_h} h — "
            f"windage α={run.windage}, K_h={run.k_h} m²/s, field: {run.field_source}",
            68_000,
            n_frames=len(run_density.snapshots),
            horizon_h=run.horizon_h,
        )
    )
    events.append(
        _event(
            "FUSING",
            0.84,
            f"Scoring {n_vessels} AIS-visible vessels against the reconstructed origin",
            76_000,
            n_vessels=n_vessels,
            n_ranked=len(result.candidates),
            n_unrankable=len(result.unrankable),
        )
    )

    if result.candidates:
        top = result.candidates[0]
        events.append(
            _event(
                "FUSING",
                0.94,
                f"Top candidate MMSI {top.mmsi} ({names.get(top.mmsi, '')}) — "
                f"LR {top.lr:.1f}, log LR {top.log_lr:.2f} nats",
                88_000,
                mmsi=top.mmsi,
                lr=round(top.lr, 3),
                log_lr=round(top.log_lr, 4),
                terms_nats={
                    k: None if v is None else round(v, 4) for k, v in top.terms.items()
                },
            )
        )

    culprit = result.culprit
    if culprit is None:
        message = (
            f"UNATTRIBUTED — no vessel reached LR 10 across {n_vessels} in frame; "
            "case queued for cross-check"
        )
    else:
        message = (
            f"{result.verdict} — MMSI {culprit.mmsi} ({names.get(culprit.mmsi, '')}), "
            f"LR {culprit.lr:.1f}, posterior {culprit.posterior:.2f}"
        )
    done: dict[str, Any] = {
        "verdict": result.verdict,
        "none_of_the_above_posterior": round(result.none_posterior, 4),
        "n_vessels_in_frame": n_vessels,
    }
    if culprit is None:
        done["culprit_mmsi"] = None
        done["unavailable"] = {
            "culprit_mmsi": "no vessel reached LR 10; UNATTRIBUTED never names one (§5.4)"
        }
    else:
        done["culprit_mmsi"] = culprit.mmsi
    events.append(_event("DONE", 1.0, message, 100_000, **done))
    return events


def build_sc02(root: Path) -> dict[str, Any]:
    """The Look-alike Trap — P0-CRITICAL. It proves the system refuses to accuse."""
    code = "SC-02"
    directory = root / code
    frame = solver_mod.Frame.for_point(SC02_CENTRE_LON, SC02_CENTRE_LAT)
    centre_m = frame.forward(np.array([SC02_CENTRE_LON]), np.array([SC02_CENTRE_LAT]))[0]

    # A low-wind look-alike: broad, rounded and amorphous, not the linear smear a
    # discharge-while-underway leaves. The shape is authored; what matters for this
    # scenario is the wind gate, which is evaluated for real against config.
    rng = np.random.default_rng(SEED_SC02)
    angles = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    radii = SC02_RADIUS_M * (1.0 + 0.22 * np.sin(3.0 * angles) + 0.08 * rng.standard_normal(40))
    ring = np.stack(
        (centre_m[0] + radii * np.cos(angles), centre_m[1] + radii * np.sin(angles)), axis=-1
    )
    formation_utm = Polygon(ring).buffer(0)

    specs = list(OTHER_VESSELS[:12])
    tracks, serialised_tracks = _build_frame_tracks(
        specs, frame, SC02_T0, centre_m, SC02_T0, 0.0, gaps={}
    )

    scene = _scene(
        code,
        "The Look-alike Trap",
        SC02_T0,
        SC02_CENTRE_LON,
        SC02_CENTRE_LAT,
        SC02_WIND_MS,
        wind_dir_deg=112.0,
        frame=frame,
    )
    detection = _detection(
        f"{code}-D1",
        code,
        formation_utm,
        frame,
        SC02_WIND_MS,
        _count_ships_within(tracks, centre_m, 20_000.0),
    )
    detection["provenance"] = (
        "Authored low-wind dark formation; no segmenter ran (§15). Features and "
        "classification are engine output; the polygon is not."
    )

    gate_reason = (
        f"wind gate violated ({SC02_WIND_MS} m/s < {settings.wind_gate_min_ms} m/s): below "
        f"{settings.wind_gate_min_ms} m/s the sea surface itself mimics oil, so the dark "
        f"formation is not trustworthy evidence of a slick (§5.2). It classifies "
        f"{detection['class']} at P(oil) {detection['p_oil']:.2f} — a rule-based score "
        "over the §5.2 physics, not a trained-model output — and the gate alone would "
        "withhold attribution regardless of what that number said."
    )
    attribution_refusal = {
        "issued": False,
        "verdict": "UNATTRIBUTED",
        "reason": gate_reason,
        "recommendation": (
            "Queue for cross-check against a second pass in a wind window of "
            f"{settings.wind_gate_min_ms}–{settings.wind_gate_max_ms} m/s."
        ),
        "candidates": [],
        "n_vessels_in_frame": len(specs),
    }
    detection["attribution"] = attribution_refusal

    _write_json(directory / "scene.json", scene)
    _write_json(directory / "detection.json", detection)
    _write_json(directory / "ais.json", _serialise_ais(specs, tracks, serialised_tracks))

    events = [
        _event(
            "SEGMENTING",
            0.10,
            f"Loading scene {scene['id']} — {SC02_T0:%Y-%m-%d %H:%MZ}, wind {SC02_WIND_MS} m/s",
            0,
            scene_id=code,
            badges=scene["badges"],
        ),
        _event(
            "SEGMENTING",
            0.35,
            f"1 dark formation delineated — {detection['features']['area_km2']:.1f} km², "
            f"shape complexity {detection['features']['shape_complexity']:.2f}",
            8_000,
            detection_id=detection["id"],
            provisional=True,
        ),
        _event(
            "DISCRIMINATING",
            0.70,
            f"WIND GATE VIOLATED — {SC02_WIND_MS} m/s is below the "
            f"{settings.wind_gate_min_ms} m/s floor; at this wind the sea surface mimics oil",
            20_000,
            wind_gate_violated=True,
            wind_speed_ms=SC02_WIND_MS,
            wind_gate_range_ms=[settings.wind_gate_min_ms, settings.wind_gate_max_ms],
            p_oil=detection["p_oil"],
            p_class=detection["class"],
            method=detection["method"],
            note=detection["note"],
            factors=detection["shap_factors"],
            unavailable=detection["unavailable"],
        ),
        _event(
            "DONE",
            1.0,
            "No attribution issued — the detection is not trustworthy evidence, "
            "so no vessel is named. Case queued for cross-check.",
            34_000,
            verdict="UNATTRIBUTED",
            attribution_issued=False,
            reason=gate_reason,
            n_vessels_in_frame=len(specs),
        ),
    ]
    _write_json(
        directory / "ws.json",
        {
            "scenario": code,
            "timings_ms": {},
            "timings_note": "No drift or attribution ran; there is nothing to time.",
            "total_playback_ms": events[-1]["playback_offset_ms"],
            "events": events,
        },
    )

    return {
        "code": code,
        "name": "The Look-alike Trap",
        "verdict": "UNATTRIBUTED",
        "attribution_issued": False,
        "wind_gate_violated": True,
        "n_detections": 1,
        "n_vessels_in_frame": len(specs),
        "artefacts": sorted(p.stem for p in directory.glob("*.json")),
    }


def build_sc03(root: Path) -> dict[str, Any]:
    """Clean Sea — zero detections. Proves no false positives."""
    code = "SC-03"
    directory = root / code
    frame = solver_mod.Frame.for_point(SC03_CENTRE_LON, SC03_CENTRE_LAT)

    scene = _scene(
        code,
        "Clean Sea",
        SC03_T0,
        SC03_CENTRE_LON,
        SC03_CENTRE_LAT,
        SC03_WIND_MS,
        wind_dir_deg=248.0,
        frame=frame,
    )
    scene["detections"] = []
    scene["n_detections"] = 0
    _write_json(directory / "scene.json", scene)

    events = [
        _event(
            "SEGMENTING",
            0.30,
            f"Loading scene {scene['id']} — {SC03_T0:%Y-%m-%d %H:%MZ}, wind {SC03_WIND_MS} m/s "
            f"(inside the {settings.wind_gate_min_ms}–{settings.wind_gate_max_ms} m/s gate)",
            0,
            scene_id=code,
            badges=scene["badges"],
        ),
        _event(
            "SEGMENTING",
            0.75,
            "No dark formations above threshold — 0 detections",
            5_000,
            n_detections=0,
        ),
        _event(
            "DONE",
            1.0,
            "Clean sea. Nothing to rewind, nothing to attribute.",
            11_000,
            n_detections=0,
            attribution_issued=False,
        ),
    ]
    _write_json(
        directory / "ws.json",
        {
            "scenario": code,
            "timings_ms": {},
            "timings_note": "No detection, drift or attribution ran; there is nothing to time.",
            "total_playback_ms": events[-1]["playback_offset_ms"],
            "events": events,
        },
    )

    return {
        "code": code,
        "name": "Clean Sea",
        "n_detections": 0,
        "artefacts": sorted(p.stem for p in directory.glob("*.json")),
    }


def build_scenario(scenario_code: str) -> dict[str, Any]:
    root = Path(settings.fixtures_dir)
    builders = {"SC-01": build_sc01, "SC-02": build_sc02, "SC-03": build_sc03}
    if scenario_code not in builders:
        raise ValueError(f"Unknown scenario {scenario_code!r}; expected one of {sorted(builders)}")
    return builders[scenario_code](root)


def main() -> int:
    root = Path(settings.fixtures_dir)
    root.mkdir(parents=True, exist_ok=True)

    summaries = [build_scenario(code) for code in ("SC-01", "SC-02", "SC-03")]

    _write_json(
        root / "manifest.json",
        {
            "generator": {
                "script": "scripts/build_fixtures.py",
                "version": GENERATOR_VERSION,
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "git_commit": _git_commit(),
            },
            "provisional": True,
            "frozen": False,
            "why_provisional": (
                "§15 specifies fixtures from real engine runs on the real Part III "
                "scenes. Zenodo Part III was unavailable at build time and data/scenes "
                "and data/ais are empty, so the drift physics and the LR fusion are real "
                "but the slick geometry and the whole AIS frame are authored. This set "
                "is not the §9-frozen one: regenerate and freeze when Tier 1 or Tier 2 "
                "data lands."
            ),
            "data_tier": "none — no SAR raster in this build; §0 Tier 4 was NOT invoked",
            "badges": [AUTHORED_SCENE_BADGE, SIMULATED_AIS_BADGE],
            "pending": [
                "Real SAR scenes (Zenodo Part III holdout 146–150, or §0 Tier 2 fallback)",
                "Real Danish AIS background traffic — every track here is injected",
                "Segmentation: the slick polygons are authored, not detected",
                "A trained LightGBM discriminator: p_oil and class here come from the "
                "rule-based §5.2 physics scorer, labelled method=rule_based",
                "SHAP factors: the factor breakdown is rule-based, basis=rule_based, "
                "not SHAP over a trained model",
                "Radiometric and texture features: no pixels to measure",
                "distance_to_coast_km: no coastline dataset bundled",
            ],
            "engine": {
                "drift_solver": "real (§5.1) — RK4 Lagrangian, forward and backward",
                "attribution": "real (§5.3, §5.4) — E1–E4, priors, LR fusion, decision bands",
                "field_source_chain": "§5.1 tier chain; this build resolved to the synthetic field",
            },
            "scenarios": summaries,
        },
    )

    print()
    for summary in summaries:
        print(f"  {summary['code']}  {summary['name']}")
        for key, value in summary.items():
            if key not in ("code", "name", "artefacts"):
                print(f"      {key}: {value}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
