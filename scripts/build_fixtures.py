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

§0 TIER 4 IS ELECTED, FOR SC-01'S RASTER ONLY
---------------------------------------------
SC-01 renders a synthetic Sigma0 raster (scripts/synth_sar.py) so that the two
§5.2 pixel-derived rule terms - contrast_db and edge_gradient_mean - have
something real to measure. That is synthetic SAR, which §0 calls Tier 4 and
requires be reached by an explicit human decision rather than by a code path
degrading into it. It was decided explicitly; nothing here falls back into it,
and SC-02 and SC-03 have no raster at all.

The raster is scenario imagery for the demo. It is NOT evidence that detection
works on real SAR, and no detection metric may be quoted from it (§16). It
carries `SYNTHETIC - NOT SAR IMAGERY` on the scene, on every feature derived from
it, in the WS stage events and in the manifest.

PLACEMENT IS GUARDED
--------------------
An earlier run of this script put SC-01 on Bornholm: 80 of 116 slick vertices,
the release point and ~90% of the particles at the worst snapshot were on the
island, and every number downstream was arithmetically correct and physically
meaningless. scripts/land_guard.py now refuses to let that be written - it
raises, the build stops, and nothing is emitted.

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
from collections.abc import Sequence
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
from app.attribution import gate as gate_mod
from app.attribution import priors as priors_mod
from app.config import settings
from app.detection import discriminator as discriminator_mod
from app.detection import explain as explain_mod
from app.detection import features as features_mod
from app.drift import density as density_mod
from app.drift import fields as fields_mod
from app.drift import solver as solver_mod
from scripts import land_guard, synth_sar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_fixtures")

GENERATOR_VERSION = 1

# --------------------------------------------------------------- authored ----
# Everything in this block is a human-authored scenario parameter, not a
# measurement. It is listed here, in one place, so that a reader can see exactly
# what was assumed and what was computed from it.

# Eastern Bornholm Basin, open water, inside Danish AIS coverage.
#
# This was 15.00 / 55.20 and that was wrong: it sits 0.8 km off Bornholm, and the
# drift carried the slick straight onto the island - 80 of 116 vertices and ~90%
# of the particles ashore. Measured here with the real solver: release over
# water, 0 of 111 slick vertices on land, no particles on land at any snapshot,
# 53 km of clearance. land_guard enforces it rather than trusting this comment.
AOI_LON = 16.30
AOI_LAT = 55.55

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
# SC-01 alone renders pixels (§0 Tier 4, elected). The badge is owned by the
# module that makes them, so the label cannot drift from the thing it labels.
SYNTHETIC_RASTER_BADGE = synth_sar.BADGE
GATE_PASSED = "the gate passed; there is nothing withheld and no refusal to state"
SYNTHETIC_PIXELS = (
    "measured from the SC-01 synthetic Sigma0 raster — §0 Tier 4 scenario "
    "imagery, not real SAR; no detection metric may be quoted from it (§16)"
)

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
    raster: synth_sar.SyntheticScene | None = None,
    raster_name: str | None = None,
) -> dict[str, Any]:
    """One scene block.

    With `raster` given the scene carries synthetic pixels (§0 Tier 4, elected
    for SC-01 only) and the badge that says so. Spacing and incidence then have
    values - but they are properties of an authored product, not measurements of
    a real one, so they are labelled `authored` rather than quietly promoted to
    looking surveyed.
    """
    centre_m = frame.forward(np.array([centre_lon]), np.array([centre_lat]))[0]
    box = shapely.box(
        centre_m[0] - SCENE_HALF_SPAN_M,
        centre_m[1] - SCENE_HALF_SPAN_M,
        centre_m[0] + SCENE_HALF_SPAN_M,
        centre_m[1] + SCENE_HALF_SPAN_M,
    )
    scene: dict[str, Any] = {
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

    if raster is None:
        return scene

    scene["sensor"] = "Sentinel-1 geometry, synthetic pixels (§0 Tier 4, elected)"
    scene["incidence_angle_deg"] = raster.incidence_angle_deg
    scene["pixel_spacing_m"] = raster.pixel_spacing_m
    scene["raster_path"] = raster_name
    scene["raster_shape_px"] = list(raster.shape)
    scene["raster_crs"] = f"EPSG:{raster.crs_epsg}"
    scene["raster_units"] = "dB (Sigma0 VV), stored as int16 hundredths"
    scene["raster_method"] = raster.method
    scene["badges"] = [SYNTHETIC_RASTER_BADGE]
    scene["unavailable"] = {}
    scene["provenance"] = {
        "incidence_angle_deg": "authored product geometry, not a measured one",
        "pixel_spacing_m": "authored product geometry, not a measured one",
        "raster_path": SYNTHETIC_PIXELS,
    }
    return scene


def _detection(
    detection_id: str,
    scene_id: str,
    polygon_utm: Polygon,
    frame: solver_mod.Frame,
    wind_speed_ms: float,
    n_ships_within_20km: int | None,
    raster: synth_sar.SyntheticScene | None = None,
) -> tuple[dict[str, Any], discriminator_mod.DiscriminatorResult]:
    """One detection block: real features, a real classification, honest gaps.

    The polygon is authored — no segmenter ran (§15) — but everything computed
    FROM it is engine output. `features.extract` produces the §5.2 vector and
    marks what it could not measure, and `discriminator.score` classifies it.
    With no trained model on disk that classification comes from the rule-based
    §5.2 physics scorer, and the block says so in `method` and `note` rather
    than letting a reader assume a model ran.

    Given a `raster` the eight image-derived features resolve instead of coming
    back None — SC-01 only, §0 Tier 4. They are measured for real, by the same
    `features.extract` a live scene would use, from pixels that are synthetic.
    Both halves of that sentence travel with them in `feature_provenance`.

    Returns the payload and the `DiscriminatorResult` behind it. The result is
    what `attribution.gate` reads: the caller must not re-derive a class from the
    payload's rounded `p_oil`, because a detection sitting on the threshold would
    then be classified twice, by two different numbers.
    """
    geometry = _hull_geometry(polygon_utm)
    extracted = features_mod.extract(
        polygon_utm,
        sigma0_db=None if raster is None else raster.sigma0_db,
        transform=None if raster is None else raster.transform,
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

    provenance = {name: AUTHORED_POLYGON for name in geometry}
    if raster is not None:
        provenance.update(
            {
                name: SYNTHETIC_PIXELS
                for name in features_mod.RADIOMETRIC_FEATURES + features_mod.TEXTURE_FEATURES
                if values.get(name) is not None
            }
        )

    payload = {
        "id": detection_id,
        "scene_id": scene_id,
        "geom": _geojson(polygon_utm, frame),
        "features": {
            name: (round(value, 4) if isinstance(value, float) else value)
            for name, value in values.items()
        },
        "feature_provenance": provenance,
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
    return payload, result


def _guard_particles(
    mask: land_guard.LandMask,
    runs: Sequence[tuple[str, solver_mod.DriftRun]],
) -> dict[str, Any]:
    """Every particle of every snapshot of every run, against the land mask.

    Runs arrive already labelled rather than keyed by a parameter, so the forward
    leg - which has no windage variant - cannot end up described by whatever
    number happened to key it.

    Returns a summary rather than one report per snapshot: there are ~75 across
    the three windage variants and the interesting one is the worst. The guard has
    already raised by the time this returns, so what is recorded here is the
    margin, not the verdict.
    """
    worst: land_guard.PlacementReport | None = None
    n_checked = 0
    for label, run in runs:
        for state in run.states:
            lon, lat = run.frame.inverse(state.positions)
            report = land_guard.check_points(
                mask,
                f"{label} t{state.t_offset_min:+d} min particles",
                lon,
                lat,
                max_fraction_on_land=land_guard.MAX_PARTICLE_LAND_FRACTION,
            )
            n_checked += 1
            if worst is None or report.fraction_on_land > worst.fraction_on_land:
                worst = report
    if worst is None:
        raise land_guard.LandPlacementError("no drift snapshots to check.")
    return {
        "snapshots_checked": n_checked,
        "max_fraction_on_land": round(worst.fraction_on_land, 6),
        "max_fraction_at": worst.label,
        "tolerance": land_guard.MAX_PARTICLE_LAND_FRACTION,
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

    # --- placement guard, before a single particle is integrated ---------------
    # This scenario was previously authored 0.8 km off Bornholm and drifted onto
    # it. Nothing downstream noticed, because nothing downstream has an opinion
    # about land. Checked here first so a misplaced release fails in a second
    # rather than after two drift legs.
    mask = land_guard.load_mask()
    release_lonlat = authoring.to_wgs84.transform(
        float(release_end_m[0]), float(release_end_m[1])
    )
    placement: dict[str, Any] = {
        "mask_source": mask.source,
        "mask_aoi": {
            "lon": [mask.lon_min, mask.lon_max],
            "lat": [mask.lat_min, mask.lat_max],
        },
        "release_point": land_guard.check_points(
            mask,
            f"{code} release point",
            np.array([release_lonlat[0]]),
            np.array([release_lonlat[1]]),
            clearance_of=Point(release_lonlat),
        ).as_dict(),
        "release_corridor": land_guard.check_polygon(
            mask, f"{code} release corridor", corridor_wgs84
        ).as_dict(),
    }

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
    placement["observed_slick"] = land_guard.check_polygon(
        mask, f"{code} observed slick", observed_wgs84
    ).as_dict()

    # --- backward leg: what the pipeline actually sees --------------------------
    runs: dict[float, solver_mod.DriftRun] = {}
    densities: dict[float, density_mod.RunDensity] = {}
    for windage in WINDAGE_VARIANTS:
        run = solver_mod.run(
            observed_wgs84, "backward", field, t0=T0, windage=windage, seed=SEED_SC01
        )
        runs[windage] = run
        densities[windage] = density_mod.compute_run_density(run.states)

    placement["forward_particles"] = _guard_particles(
        mask, [(f"{code} forward leg", forward_run)]
    )
    placement["backward_particles"] = _guard_particles(
        mask,
        [(f"{code} backward α={windage:.3f}", run) for windage, run in sorted(runs.items())],
    )

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

    slick_centroid_m = np.array(
        _to_utm(observed_wgs84, default_run.frame).centroid.coords[0]
    )
    release_centroid_run_m = np.array(
        _to_utm(corridor_wgs84, default_run.frame).centroid.coords[0]
    )
    observed_run_utm = _to_utm(observed_wgs84, default_run.frame)

    # --- the scene, with synthetic pixels (§0 Tier 4, elected for SC-01) --------
    # Rendered from the slick polygon and the run's own wind field, so the two
    # §5.2 pixel terms have something real to measure. Scenario imagery, badged
    # as such everywhere it surfaces; not evidence about real SAR (§16).
    wind_speed_ms, wind_dir_deg = _wind_from_field(field, AOI_LON, AOI_LAT, T0)
    scene_centre_m = default_run.frame.forward(np.array([AOI_LON]), np.array([AOI_LAT]))[0]
    raster = synth_sar.render(
        observed_run_utm,
        scene_centre_m,
        int(default_run.frame.crs.to_epsg()),
        field,
        default_run.frame.to_wgs84,
        T0,
        seed=SEED_SC01,
    )
    raster_name = "sigma0_vv_db.tif"
    raster_bytes = synth_sar.write_geotiff(raster, directory / raster_name)

    scene = _scene(
        code,
        "Baltic Night Discharge",
        T0,
        AOI_LON,
        AOI_LAT,
        wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
        frame=default_run.frame,
        raster=raster,
        raster_name=raster_name,
    )
    detection, discriminator_result = _detection(
        f"{code}-D1",
        code,
        observed_run_utm,
        default_run.frame,
        wind_speed_ms,
        _count_ships_within(tracks, slick_centroid_m, 20_000.0),
        raster=raster,
    )
    detection["provenance"] = (
        "Alpha hull of a real 6 h forward drift run seeded on the authored release "
        "corridor. The attribution engine sees only this polygon. Features and "
        "classification are engine output; the polygon is not."
    )

    # --- the gate: may this detection name a vessel at all? ---------------------
    # SC-01 previously classified look-alike and attributed a culprit anyway.
    # Attribution does not run unless the detection reads as oil (§5.2 wind gate,
    # same principle). A blocked frame writes the refusal and stops.
    gate_decision = gate_mod.evaluate(
        discriminator_result,
        detection["wind_gate_violated"],
        detection["wind_gate_reason"],
    )

    # --- score and fuse, once per windage variant --------------------------------
    results: dict[float, fusion_mod.FrameResult] = {}
    fuse_ms = 0
    if gate_decision.passed:
        fuse_started = datetime.now(tz=UTC)
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
    else:
        logger.warning(
            "%s: attribution withheld by the gate (%s) — %s",
            code,
            gate_decision.code,
            gate_decision.reason,
        )
        detection["attribution"] = gate_decision.as_refusal(len(specs))

    default_result = results.get(DEFAULT_WINDAGE)

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
        if windage in results:
            _write_json(
                directory / f"attribution{suffix}.json",
                _serialise_attribution(results[windage], names, len(specs), windage),
            )

    culprit_channels = (
        None
        if default_result is None
        else next(
            (
                entry.channels
                for entry in (*default_result.candidates, *default_result.unrankable)
                if entry.mmsi == CULPRIT_MMSI
            ),
            None,
        )
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
            "placement": placement,
            "placement_note": (
                "Measured against data/geo/baltic_land_mask.geojson before anything "
                "was written. A scenario whose oil sits on land is not a scenario, "
                "and scripts/land_guard.py raises rather than emitting one."
            ),
            "attribution_gate": {
                "passed": gate_decision.passed,
                "blocked_by": gate_decision.code,
                "reason": gate_decision.reason or None,
                "note": (
                    "Attribution runs only for a detection that reads as oil. A "
                    "look-alike does not name a culprit."
                ),
                "unavailable": (
                    {}
                    if not gate_decision.passed
                    else {
                        "blocked_by": GATE_PASSED,
                        "reason": GATE_PASSED,
                    }
                ),
            },
            "identified_culprit_mmsi": (
                None
                if default_result is None or default_result.culprit is None
                else default_result.culprit.mmsi
            ),
            "identified_correctly": (
                default_result is not None
                and default_result.culprit is not None
                and default_result.culprit.mmsi == CULPRIT_MMSI
            ),
            "verdict": "UNATTRIBUTED" if default_result is None else default_result.verdict,
        },
    )

    events = _ws_sc01(
        scene,
        detection,
        geometry,
        default_run,
        default_density,
        default_result,
        names,
        len(specs),
        gate_decision,
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

    culprit = None if default_result is None else default_result.culprit
    return {
        "code": code,
        "name": "Baltic Night Discharge",
        "verdict": "UNATTRIBUTED" if default_result is None else default_result.verdict,
        "attribution_issued": gate_decision.passed,
        "attribution_blocked_by": gate_decision.code,
        "p_oil": detection["p_oil"],
        "detection_class": detection["class"],
        "culprit_mmsi": None if culprit is None else culprit.mmsi,
        "lr": None if culprit is None else round(culprit.lr, 2),
        "n_vessels_in_frame": len(specs),
        "n_ranked": 0 if default_result is None else len(default_result.candidates),
        "origin_error_km": recovery["origin_error_km"],
        "t_star_error_min": recovery["t_star_error_min"],
        "slick_clearance_km": placement["observed_slick"]["clearance_km"],
        "raster": raster_name,
        "raster_bytes": raster_bytes,
        "raster_badge": SYNTHETIC_RASTER_BADGE,
        "unavailable": (
            {"attribution_blocked_by": GATE_PASSED}
            if gate_decision.passed
            else {"culprit_mmsi": gate_decision.reason, "lr": gate_decision.reason}
        ),
        "artefacts": sorted(p.stem for p in directory.glob("*.json")),
    }


def _ws_sc01(
    scene: dict[str, Any],
    detection: dict[str, Any],
    geometry: dict[str, float],
    run: solver_mod.DriftRun,
    run_density: density_mod.RunDensity,
    result: fusion_mod.FrameResult | None,
    names: dict[int, str],
    n_vessels: int,
    gate_decision: gate_mod.GateDecision,
) -> list[dict[str, Any]]:
    """§12 budgets SC-01 at 110 s. Messages carry numbers this run produced.

    `result` is None when the gate withheld attribution. The sequence then stops
    after DISCRIMINATING with a refusal, because that is what actually happened —
    a rewind whose conclusion is discarded is theatre.
    """
    events = [
        _event(
            "SEGMENTING",
            0.05,
            f"Loading scene {scene['id']} — eastern Bornholm Basin, {T0:%Y-%m-%d %H:%MZ}",
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
            badges=scene["badges"],
            pixel_features_note=SYNTHETIC_PIXELS,
            unavailable=detection["unavailable"],
        ),
    ]

    if result is None:
        events.append(
            _event(
                "DONE",
                1.0,
                "No attribution issued — the detection does not read as oil, so no "
                "vessel is named. Case queued for cross-check.",
                26_000,
                verdict="UNATTRIBUTED",
                attribution_issued=False,
                blocked_by=gate_decision.code,
                reason=gate_decision.reason,
                recommendation=gate_decision.recommendation,
                n_vessels_in_frame=n_vessels,
            )
        )
        return events

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

    # Same guard as SC-01. A look-alike sitting on land is no more shippable than
    # a slick that does; the wind-gate refusal is only meaningful over water.
    mask = land_guard.load_mask()
    placement = {
        "mask_source": mask.source,
        "dark_formation": land_guard.check_polygon(
            mask, f"{code} dark formation", _to_wgs84(formation_utm, frame)
        ).as_dict(),
    }

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
    detection, discriminator_result = _detection(
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

    # The refusal is engine output, not prose written here. It used to be a
    # hand-assembled dict, which meant SC-02's honesty lived in this file rather
    # than in the pipeline and covered exactly one scenario (§2.2).
    gate_decision = gate_mod.evaluate(
        discriminator_result,
        detection["wind_gate_violated"],
        detection["wind_gate_reason"],
    )
    if gate_decision.passed:
        raise RuntimeError(
            f"{code} is the look-alike trap and must not attribute, but the gate "
            f"passed it: class={detection['class']} p_oil={detection['p_oil']} "
            f"wind_gate_violated={detection['wind_gate_violated']}. Refusing to "
            "write a scenario that contradicts its own purpose (§12 P0-CRITICAL)."
        )
    attribution_refusal = gate_decision.as_refusal(len(specs))
    gate_reason = gate_decision.reason
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
            blocked_by=gate_decision.code,
            reason=gate_reason,
            recommendation=gate_decision.recommendation,
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
        "attribution_blocked_by": gate_decision.code,
        "wind_gate_violated": True,
        "p_oil": detection["p_oil"],
        "detection_class": detection["class"],
        "n_detections": 1,
        "n_vessels_in_frame": len(specs),
        "formation_clearance_km": placement["dark_formation"]["clearance_km"],
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
            "data_tier": (
                "§0 Tier 4 ELECTED for SC-01's raster only — synthetic Sigma0 pixels, "
                "rendered from the authored slick polygon by scripts/synth_sar.py. An "
                "explicit human decision, not a fallback: no code path degrades into "
                "it, and SC-02 and SC-03 carry no raster at all. Scenario imagery for "
                "the demo — NOT evidence that detection works on real SAR, and no "
                "detection metric may be quoted from it (§16)."
            ),
            "tier_4": {
                "elected": True,
                "scope": "SC-01 raster only",
                "badge": SYNTHETIC_RASTER_BADGE,
                "generator": "scripts/synth_sar.py",
                "method_note": (
                    "Dark-region geometry is the slick polygon itself; damping is the "
                    "midpoint of the 3–10 dB range CLAUDE.md states for mineral oil; "
                    "edge sharpness is one resolution cell; speckle is Gamma at an ENL "
                    "derived from the IW GRDH product chain; background is the drift "
                    "run's own wind field. A method, not a set of knobs."
                ),
                "no_metric_from_it": (
                    "§16 — this raster establishes nothing about detection accuracy. It "
                    "exists so the two §5.2 pixel terms resolve."
                ),
            },
            "land_guard": {
                "module": "scripts/land_guard.py",
                "mask": "data/geo/baltic_land_mask.geojson",
                "particle_tolerance": land_guard.MAX_PARTICLE_LAND_FRACTION,
                "note": (
                    "Slick vertices, release point and every drift snapshot are checked "
                    "against a real land polygon before anything is written. SC-01 was "
                    "previously emitted onto Bornholm with ~90% of its particles ashore; "
                    "the guard raises rather than warns, so it cannot happen quietly."
                ),
            },
            "badges": [AUTHORED_SCENE_BADGE, SIMULATED_AIS_BADGE, SYNTHETIC_RASTER_BADGE],
            "pending": [
                "Real SAR scenes (Zenodo Part III holdout 146–150, or §0 Tier 2 fallback)",
                "Real Danish AIS background traffic — every track here is injected",
                "Segmentation: the slick polygons are authored, not detected",
                "A trained LightGBM discriminator: p_oil and class here come from the "
                "rule-based §5.2 physics scorer, labelled method=rule_based",
                "Real SAR pixels. SC-01's raster is synthetic (§0 Tier 4, elected) and "
                "SC-02/SC-03 have none, so no detection metric is quotable anywhere in "
                "this set (§16)",
                "SHAP factors: the factor breakdown is rule-based, basis=rule_based, "
                "not SHAP over a trained model",
                "Radiometric and texture features on SC-02: no pixels to measure there",
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
            if key not in ("code", "name", "artefacts", "unavailable"):
                print(f"      {key}: {value}")
    print()

    # SC-01 exists to attribute. If the gate withheld it the set is still honest,
    # but it is not the demo §12 describes, and that must not be discovered by
    # someone scrolling back through the log.
    sc01 = next(entry for entry in summaries if entry["code"] == "SC-01")
    if not sc01["attribution_issued"]:
        print("  " + "!" * 68)
        print(f"  SC-01 DID NOT ATTRIBUTE — blocked by: {sc01['attribution_blocked_by']}")
        print(f"  class={sc01['detection_class']} p_oil={sc01['p_oil']}")
        print("  The gate is working; the scenario is not. Fix the physics, not the")
        print("  weights (§2.2) — a scorer tuned to reproduce a target is the thing")
        print("  the whole set exists to avoid.")
        print("  " + "!" * 68)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
