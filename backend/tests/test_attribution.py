"""Attribution channels, priors and LR fusion — §5.3, §5.4.

The tests that matter here are the ones about refusal, not the ones about
ranking. A system that names a culprit is easy; a system that declines to name
one when the evidence does not reach ln(10), and that says which channel had
nothing to say and why, is the thing being built (§2.4).

Three properties are load-bearing and each has a test that fails loudly:

  * E2 and E3 never touch the current field. Enforced structurally on the import
    graph, and demonstrated behaviourally against two different ocean states.
  * No sub-threshold candidate is ever promoted to a culprit (§5.4, §9).
  * A vessel with no evidence is not ranked on its prior alone.
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import app
from app.attribution import channels as C
from app.attribution import fusion as FU
from app.attribution import priors as P

T0 = datetime(2024, 3, 1, 2, 0, tzinfo=UTC)

# --- the synthetic frame ------------------------------------------------------
# A 60 km x 60 km run grid at 200 m, 25 snapshots at 30 min: the shape a 12 h
# backward run emits. The cloud rewinds along a straight path and spreads as it
# goes, which is what diffusion does in both time directions (§5.1).

CELL_M = 200.0
N_CELLS = 300
N_SNAPSHOTS = 25
T_STAR_INDEX = 13  # T-6h30m — where the culprit crossed

SLICK_ORIENTATION_DEG = 60.0
MAJOR_AXIS_KM = 22.2
STRETCH_AT_T_STAR = 1.5
CULPRIT_MMSI = 219000001

# 22.2 km observed / 1.5 stretch = 14.8 km released; at 12 kn that is a ~40 min
# discharge, which is where the §5.3 lognormal peaks.
CULPRIT_SOG_KN = 12.0
CULPRIT_COG_DEG = SLICK_ORIENTATION_DEG


@dataclass
class FakeGrid:
    x0_m: float
    y0_m: float
    cell_size_m: float
    nx: int
    ny: int


@dataclass
class FakeSnapshot:
    t_offset_min: int
    probability: np.ndarray
    stretch_factor: float


@dataclass
class FakeDensity:
    grid: FakeGrid
    snapshots: list[FakeSnapshot] = field(default_factory=list)


def cloud_centre(k: int) -> np.ndarray:
    return np.array([32000.0 - 700.0 * k, 34000.0 - 400.0 * k])


def build_density() -> FakeDensity:
    """A rewinding, spreading cloud, normalised to sum to 1 over the whole run.

    Run-wide rather than per-snapshot normalisation, exactly as §5.1 specifies,
    because E1 integrates mass across snapshots.
    """
    grid = FakeGrid(x0_m=0.0, y0_m=0.0, cell_size_m=CELL_M, nx=N_CELLS, ny=N_CELLS)
    xs = grid.x0_m + (np.arange(grid.nx) + 0.5) * CELL_M
    ys = grid.y0_m + (np.arange(grid.ny) + 0.5) * CELL_M

    raw = []
    for k in range(N_SNAPSHOTS):
        centre = cloud_centre(k)
        sigma = 800.0 + 70.0 * k
        dx = (xs - centre[0])[None, :]
        dy = (ys - centre[1])[:, None]
        raw.append(np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2)))

    total = sum(float(f.sum()) for f in raw)
    return FakeDensity(
        grid=grid,
        snapshots=[
            FakeSnapshot(
                t_offset_min=-30 * k,
                probability=f / total,
                stretch_factor=1.0 + k * (STRETCH_AT_T_STAR - 1.0) / T_STAR_INDEX,
            )
            for k, f in enumerate(raw)
        ],
    )


def ais_times(step_min: int = 10) -> list[datetime]:
    """Samples spanning the whole 12 h rewind window plus an hour either side."""
    start = T0 - timedelta(hours=13)
    n = int(14 * 60 / step_min) + 1
    return [start + timedelta(minutes=step_min * i) for i in range(n)]


def straight_track(
    mmsi: int,
    through_m: np.ndarray,
    at_time: datetime,
    cog_deg: float,
    sog_kn: float,
    gap: tuple[datetime, datetime] | None = None,
    step_min: int = 10,
) -> C.VesselTrack:
    """A constant-course, constant-speed track through a given point.

    `gap` drops every sample strictly inside the interval, which is what a vessel
    switching off its transponder looks like in the data.
    """
    heading = math.radians(cog_deg)
    direction = np.array([math.sin(heading), math.cos(heading)])
    speed_ms = sog_kn * 1852.0 / 3600.0

    times = [t for t in ais_times(step_min) if gap is None or not gap[0] < t < gap[1]]
    positions = np.array(
        [through_m + direction * speed_ms * (t - at_time).total_seconds() for t in times]
    )
    return C.VesselTrack(
        mmsi=mmsi,
        times=times,
        positions_m=positions,
        sog_kn=np.full(len(times), sog_kn),
        cog_deg=np.full(len(times), cog_deg),
    )


def perpendicular(cog_deg: float) -> np.ndarray:
    heading = math.radians(cog_deg)
    return np.array([math.cos(heading), -math.sin(heading)])


@pytest.fixture(scope="module")
def density() -> FakeDensity:
    return build_density()


@pytest.fixture(scope="module")
def culprit_frame() -> tuple[list[C.VesselTrack], dict[int, tuple[str | None, int]]]:
    """One tanker across the rewound origin, three near misses, ten bystanders.

    The near misses are deliberate: a frame where every innocent vessel is
    comfortably far away proves nothing about a method meant to run in the
    Baltic, where traffic is dense and near misses are the normal case.
    """
    t_star = T0 - timedelta(minutes=30 * T_STAR_INDEX)
    origin = cloud_centre(T_STAR_INDEX)
    perp = perpendicular(CULPRIT_COG_DEG)

    tracks = [
        straight_track(
            CULPRIT_MMSI,
            origin,
            t_star,
            CULPRIT_COG_DEG,
            CULPRIT_SOG_KN,
            gap=(t_star - timedelta(minutes=45), t_star + timedelta(minutes=45)),
        )
    ]
    vessels: dict[int, tuple[str | None, int]] = {CULPRIT_MMSI: ("Crude Oil Tanker", 1)}

    near_misses = [(3000.0, 15.0, 9.0), (-4000.0, 200.0, 14.0), (5000.0, 305.0, 6.0)]
    for i, (offset_m, cog, sog) in enumerate(near_misses):
        mmsi = 219100000 + i
        tracks.append(straight_track(mmsi, origin + perp * offset_m, t_star, cog, sog))
        vessels[mmsi] = ("Cargo", 0)

    bystanders = [
        (14000.0, 100.0, 11.0, "Cargo"),
        (-16000.0, 250.0, 13.0, "Cargo"),
        (19000.0, 340.0, 8.0, "Fishing"),
        (-21000.0, 45.0, 5.0, "Fishing"),
        (23000.0, 170.0, 17.0, "Passenger Ferry"),
        (-25000.0, 285.0, 12.0, "Cargo"),
        (27000.0, 20.0, 7.0, "Fishing"),
        (-29000.0, 130.0, 19.0, "Passenger Ferry"),
        (31000.0, 215.0, 10.0, "Cargo"),
        (-33000.0, 300.0, 4.0, "Fishing"),
    ]
    for i, (offset_m, cog, sog, vessel_type) in enumerate(bystanders):
        mmsi = 219200000 + i
        tracks.append(straight_track(mmsi, origin + perp * offset_m, t_star, cog, sog))
        vessels[mmsi] = (vessel_type, 0)

    return tracks, vessels


def fuse_frame(
    tracks: list[C.VesselTrack],
    vessels: dict[int, tuple[str | None, int]],
    density: FakeDensity,
) -> FU.FrameResult:
    scores = C.score_vessels(
        tracks,
        density,
        T0,
        slick_orientation_deg=SLICK_ORIENTATION_DEG,
        major_axis_km=MAJOR_AXIS_KM,
    )
    return FU.fuse(scores, P.frame_priors(vessels))


# ------------------------------------------------------------- VesselTrack ----


def test_position_is_linearly_interpolated_between_samples() -> None:
    times = [T0, T0 + timedelta(minutes=10)]
    track = C.VesselTrack(
        mmsi=1,
        times=times,
        positions_m=np.array([[0.0, 0.0], [1000.0, 2000.0]]),
        sog_kn=np.array([10.0, 10.0]),
        cog_deg=np.array([0.0, 0.0]),
    )
    mid = track.position_m_at(T0 + timedelta(minutes=5))
    assert mid is not None
    np.testing.assert_allclose(mid, [500.0, 1000.0])


def test_position_is_never_extrapolated_outside_the_track() -> None:
    track = straight_track(1, np.zeros(2), T0, 90.0, 10.0)
    assert track.position_m_at(track.t_end + timedelta(hours=1)) is None
    assert track.position_m_at(track.t_start - timedelta(hours=1)) is None


def test_cog_interpolates_across_the_zero_wrap() -> None:
    """350 deg to 10 deg is 20 deg apart, not 340. A linear blend gives 180."""
    times = [T0, T0 + timedelta(minutes=10)]
    track = C.VesselTrack(
        mmsi=1,
        times=times,
        positions_m=np.zeros((2, 2)),
        sog_kn=np.array([10.0, 10.0]),
        cog_deg=np.array([350.0, 10.0]),
    )
    midpoint = track.cog_deg_at(T0 + timedelta(minutes=5))
    assert midpoint is not None
    assert min(midpoint, 360.0 - midpoint) == pytest.approx(0.0, abs=1e-9)


def test_nan_sog_and_cog_read_as_unavailable() -> None:
    times = [T0, T0 + timedelta(minutes=10)]
    track = C.VesselTrack(
        mmsi=1,
        times=times,
        positions_m=np.zeros((2, 2)),
        sog_kn=np.array([np.nan, 10.0]),
        cog_deg=np.array([np.nan, 10.0]),
    )
    assert track.sog_kn_at(T0 + timedelta(minutes=5)) is None
    assert track.cog_deg_at(T0 + timedelta(minutes=5)) is None


def test_longest_gap_reports_the_full_gap_overlapping_the_window() -> None:
    times = [T0, T0 + timedelta(minutes=10), T0 + timedelta(minutes=100)]
    track = C.VesselTrack(
        mmsi=1,
        times=times,
        positions_m=np.zeros((3, 2)),
        sog_kn=np.full(3, 10.0),
        cog_deg=np.full(3, 10.0),
    )
    # The 90 min gap only just reaches into this window and is still reported whole.
    assert track.longest_gap_min(T0 + timedelta(minutes=95), T0 + timedelta(hours=4)) == 90.0
    # A window that sees only the dense stretch reports the small gap, not the big one.
    assert track.longest_gap_min(T0, T0 + timedelta(minutes=9)) == 10.0
    assert track.longest_gap_min(T0 + timedelta(hours=5), T0 + timedelta(hours=6)) == 0.0


def test_track_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="expected"):
        C.VesselTrack(
            mmsi=1,
            times=[T0, T0 + timedelta(minutes=1)],
            positions_m=np.zeros((3, 2)),
            sog_kn=np.zeros(2),
            cog_deg=np.zeros(2),
        )


# ---------------------------------------------------------------------- E1 ----


def test_e1_t_star_is_the_snapshot_contributing_maximum_mass(density: FakeDensity) -> None:
    t_star_time = T0 - timedelta(minutes=30 * T_STAR_INDEX)
    track = straight_track(
        1, cloud_centre(T_STAR_INDEX), t_star_time, CULPRIT_COG_DEG, CULPRIT_SOG_KN
    )
    mass, t_star, stretch = C.e1_mass_overlap(
        density, C.snapshot_times(T0, density), track
    )
    assert mass > 0.0
    assert t_star == t_star_time
    assert stretch == pytest.approx(STRETCH_AT_T_STAR)


def test_e1_falls_off_with_distance_from_the_origin_cloud(density: FakeDensity) -> None:
    t_star_time = T0 - timedelta(minutes=30 * T_STAR_INDEX)
    origin = cloud_centre(T_STAR_INDEX)
    perp = perpendicular(CULPRIT_COG_DEG)
    times = C.snapshot_times(T0, density)

    on_top, _, _ = C.e1_mass_overlap(
        density, times, straight_track(1, origin, t_star_time, 60.0, 12.0)
    )
    nearby, _, _ = C.e1_mass_overlap(
        density, times, straight_track(2, origin + perp * 3000.0, t_star_time, 60.0, 12.0)
    )
    far, _, _ = C.e1_mass_overlap(
        density, times, straight_track(3, origin + perp * 20000.0, t_star_time, 60.0, 12.0)
    )
    assert on_top > nearby > far


def test_e1_shares_sum_to_one_across_the_frame(
    culprit_frame: tuple[list[C.VesselTrack], dict[int, tuple[str | None, int]]],
    density: FakeDensity,
) -> None:
    tracks, _ = culprit_frame
    scores = C.score_vessels(
        tracks, density, T0, slick_orientation_deg=60.0, major_axis_km=MAJOR_AXIS_KM
    )
    assert sum(s.s1 for s in scores.values()) == pytest.approx(1.0)


def test_e1_gives_no_t_star_to_a_vessel_that_never_overlaps(density: FakeDensity) -> None:
    far = straight_track(1, np.array([200000.0, 200000.0]), T0, 90.0, 10.0)
    mass, t_star, _ = C.e1_mass_overlap(density, C.snapshot_times(T0, density), far)
    assert mass == 0.0
    assert t_star is None


# ---------------------------------------------------------------------- E2 ----


def test_e2_peaks_when_the_course_matches_the_slick_axis() -> None:
    assert C.e2_axial_coherence(60.0, 60.0) == pytest.approx(1.0)


def test_e2_folds_at_ninety_degrees() -> None:
    """A slick axis is undirected: the reciprocal heading laid the same line."""
    assert C.e2_axial_coherence(10.0, 190.0) == pytest.approx(1.0)
    assert C.e2_axial_coherence(10.0, 340.0) == C.e2_axial_coherence(10.0, 160.0)
    assert C.e2_axial_coherence(0.0, 90.0) == pytest.approx(
        math.exp(-((90.0 / 25.0) ** 2))
    )


def test_e2_is_minimised_at_a_right_angle() -> None:
    at_right_angle = C.e2_axial_coherence(0.0, 90.0)
    for cog in (0.0, 20.0, 45.0, 70.0, 110.0, 180.0):
        assert C.e2_axial_coherence(0.0, cog) >= at_right_angle


# ---------------------------------------------------------------------- E3 ----


def test_e3_peaks_at_the_lognormal_mode() -> None:
    """The mode sits at exp(mu - sigma^2), below the 90 min median."""
    from app.config import settings

    mode_min = settings.e3_tau_median_min * math.exp(-settings.e3_tau_sigma**2)
    sog_kn = MAJOR_AXIS_KM / STRETCH_AT_T_STAR / (mode_min / 60.0) / C.KNOTS_TO_KMH
    assert C.e3_kinematic_consistency(
        MAJOR_AXIS_KM, STRETCH_AT_T_STAR, sog_kn
    ) == pytest.approx(1.0)


def test_e3_shortens_the_observed_slick_by_the_stretch_factor() -> None:
    """A larger stretch means a shorter release, so a shorter implied duration."""
    slow = C.e3_kinematic_consistency(MAJOR_AXIS_KM, 1.0, 3.0)
    stretched = C.e3_kinematic_consistency(MAJOR_AXIS_KM, 4.0, 3.0)
    assert stretched > slow


def test_e3_is_zero_for_a_vessel_not_underway() -> None:
    assert C.e3_kinematic_consistency(MAJOR_AXIS_KM, STRETCH_AT_T_STAR, 0.0) == 0.0


def test_e3_never_exceeds_one() -> None:
    for sog in (0.5, 2.0, 6.0, 12.0, 18.0, 25.0, 40.0):
        assert 0.0 <= C.e3_kinematic_consistency(MAJOR_AXIS_KM, STRETCH_AT_T_STAR, sog) <= 1.0


# ---------------------------------------------------------------------- E4 ----


def test_e4_never_penalises_a_clean_transmitter() -> None:
    assert C.e4_dark_gap(0.0) == 1.0
    assert C.e4_dark_gap(-5.0) == 1.0
    assert C.e4_dark_gap(float("nan")) == 1.0


def test_e4_is_a_bounded_boost() -> None:
    assert C.e4_dark_gap(30.0) == pytest.approx(1.4)
    assert C.e4_dark_gap(90.0) == pytest.approx(2.2)
    assert C.e4_dark_gap(600.0) == pytest.approx(2.2)


def test_e4_is_monotonic_in_gap_length() -> None:
    values = [C.e4_dark_gap(g) for g in (0.0, 10.0, 30.0, 60.0, 90.0, 200.0)]
    assert values == sorted(values)


# ------------------------------------------------------------------ priors ----


def test_type_priors_match_the_spec() -> None:
    assert P.type_prior("Crude Oil Tanker") == 3.0
    assert P.type_prior("Cargo") == 1.5
    assert P.type_prior("Bulk Carrier") == 1.5
    assert P.type_prior("Fishing") == 1.0
    assert P.type_prior("Passenger Ferry") == 0.5
    assert P.type_prior("Sailing") == 1.0
    assert P.type_prior(None) == 1.0


def test_history_multiplier_is_one_plus_half_per_detection() -> None:
    assert P.history_multiplier(0) == 1.0
    assert P.history_multiplier(2) == 2.0
    assert P.prior("Tanker", 2) == pytest.approx(6.0)


def test_negative_detection_history_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        P.history_multiplier(-1)


def test_priors_are_normalised_by_the_frame_mean() -> None:
    """Mean-normalised, so a neutral vessel contributes zero nats (§5.4)."""
    normalised = P.frame_priors(
        {1: ("Tanker", 0), 2: ("Cargo", 0), 3: ("Fishing", 0), 4: ("Cargo", 0)}
    )
    assert sum(normalised.values()) / len(normalised) == pytest.approx(1.0)
    assert normalised[1] > 1.0 > normalised[3]


def test_prior_normalisation_does_not_move_with_frame_size() -> None:
    """A busier frame must not push every candidate below the §9 bands."""
    small = P.frame_priors({1: ("Tanker", 0), 2: ("Cargo", 0)})
    large = P.frame_priors({1: ("Tanker", 0), 2: ("Cargo", 0)} | {
        i: ("Cargo", 0) for i in range(3, 60)
    })
    assert large[1] == pytest.approx(small[1], rel=0.5)
    assert large[1] > 1.0


# ------------------------------------------------------------------ fusion ----


def test_decision_bands_are_the_literal_thresholds() -> None:
    assert FU.LOG_LR_MODERATE == math.log(10.0)
    assert FU.LOG_LR_STRONG == math.log(100.0)
    assert FU.NONE_OF_THE_ABOVE_LOG_LR == math.log(10.0)


def test_decision_bands_are_not_reachable_from_settings() -> None:
    """§9: a safety property, not a tunable. No env var may move them."""
    from app.config import Settings

    names = set(Settings.model_fields)
    forbidden = ("log_lr", "verdict", "decision", "strong", "moderate", "unattributed")
    assert not [n for n in names if any(word in n for word in forbidden)]


def test_verdict_boundaries_are_exact() -> None:
    assert FU.verdict_for(FU.LOG_LR_STRONG) == "STRONG"
    assert FU.verdict_for(FU.LOG_LR_MODERATE) == "MODERATE"
    assert FU.verdict_for(math.nextafter(FU.LOG_LR_STRONG, 0.0)) == "MODERATE"
    assert FU.verdict_for(math.nextafter(FU.LOG_LR_MODERATE, 0.0)) == "UNATTRIBUTED"
    assert FU.verdict_for(0.0) == "UNATTRIBUTED"


def test_background_excludes_unavailable_channels_but_not_zero_scores() -> None:
    scores = [
        C.ChannelScores(mmsi=1, s1=0.6, s2=0.8, s3=None, s4=1.0, t_star=T0),
        C.ChannelScores(mmsi=2, s1=0.4, s2=None, s3=0.2, s4=1.0, t_star=T0),
        C.ChannelScores(mmsi=3, s1=0.0, s2=0.2, s3=0.0, s4=1.0, t_star=T0),
    ]
    s1_bg, s2_bg, s3_bg = FU.background_scores(scores)
    assert s1_bg == pytest.approx(1.0 / 3.0)
    assert s2_bg == pytest.approx(0.5)
    assert s3_bg == pytest.approx(0.1)


def test_unavailable_channel_contributes_no_nats() -> None:
    with_cog = C.ChannelScores(mmsi=1, s1=0.5, s2=0.5, s3=0.5, s4=1.0, t_star=T0)
    without = C.ChannelScores(mmsi=1, s1=0.5, s2=None, s3=0.5, s4=1.0, t_star=T0)
    background = (0.5, 0.5, 0.5)
    assert FU.log_likelihood_ratio(with_cog, background, 1.0)[1]["s2"] == pytest.approx(0.0)
    assert FU.log_likelihood_ratio(without, background, 1.0)[1]["s2"] is None
    assert FU.log_likelihood_ratio(with_cog, background, 1.0)[0] == pytest.approx(
        FU.log_likelihood_ratio(without, background, 1.0)[0]
    )


def test_posteriors_and_the_none_hypothesis_sum_to_one() -> None:
    log_lrs = {1: 5.0, 2: 1.0, 3: -2.0}
    total = sum(FU.posteriors(log_lrs).values()) + FU.none_posterior(log_lrs)
    assert total == pytest.approx(1.0)


def test_none_hypothesis_dominates_a_weak_frame() -> None:
    log_lrs = {1: 0.5, 2: 0.2, 3: -1.0}
    assert FU.none_posterior(log_lrs) > max(FU.posteriors(log_lrs).values())


def test_a_lone_vessel_can_never_be_accused() -> None:
    """Every background equals its own score, so only E4 survives: max 0.79 nats.

    There is no frame to be a ratio against, and the arithmetic says so without
    anyone having to add a special case.
    """
    scores = {1: C.ChannelScores(mmsi=1, s1=1.0, s2=1.0, s3=1.0, s4=2.2, t_star=T0)}
    result = FU.fuse(scores, P.frame_priors({1: ("Tanker", 5)}))
    assert result.candidates[0].verdict == "UNATTRIBUTED"
    assert result.culprit is None


# ------------------------------------------------------- the done-conditions ----


def test_planted_culprit_ranks_first_and_is_strong(
    culprit_frame: tuple[list[C.VesselTrack], dict[int, tuple[str | None, int]]],
    density: FakeDensity,
) -> None:
    tracks, vessels = culprit_frame
    result = fuse_frame(tracks, vessels, density)

    top = result.candidates[0]
    assert top.mmsi == CULPRIT_MMSI
    assert top.rank == 1
    assert top.lr > 100.0
    assert top.verdict == "STRONG"
    assert result.culprit is not None
    assert result.culprit.mmsi == CULPRIT_MMSI
    assert result.verdict == "STRONG"

    # Every channel pulled in the same direction; none of them did it alone.
    assert all(term > 0.0 for name, term in top.terms.items() if term is not None)
    assert top.channels.t_star == T0 - timedelta(minutes=30 * T_STAR_INDEX)


def test_channel_breakdown_is_complete_for_the_ui(
    culprit_frame: tuple[list[C.VesselTrack], dict[int, tuple[str | None, int]]],
    density: FakeDensity,
) -> None:
    tracks, vessels = culprit_frame
    top = fuse_frame(tracks, vessels, density).candidates[0]
    assert set(top.terms) == {"s1", "s2", "s3", "s4", "prior"}
    assert sum(t for t in top.terms.values() if t is not None) == pytest.approx(top.log_lr)


def test_frame_with_no_true_source_is_unattributed_for_every_candidate(
    density: FakeDensity,
) -> None:
    """Traffic passing at a distance, no dark gaps, no tanker: nobody is named.

    Every vessel still gets a rank and a likelihood ratio - ranking is an
    ordering, not an accusation - but none of them clears ln(10), so the frame
    has no culprit (§5.4).
    """
    t_star_time = T0 - timedelta(minutes=30 * T_STAR_INDEX)
    origin = cloud_centre(T_STAR_INDEX)
    tracks = []
    vessels: dict[int, tuple[str | None, int]] = {}
    for i in range(12):
        bearing = math.radians(30.0 * i)
        offset = np.array([math.sin(bearing), math.cos(bearing)]) * 6000.0
        mmsi = 219300000 + i
        tracks.append(
            straight_track(mmsi, origin + offset, t_star_time, 20.0 * i, 8.0 + 0.5 * i)
        )
        vessels[mmsi] = ("Cargo", 0)

    result = fuse_frame(tracks, vessels, density)
    assert len(result.candidates) == 12
    assert all(c.verdict == "UNATTRIBUTED" for c in result.candidates)
    assert all(c.lr < 10.0 for c in result.candidates)
    assert result.culprit is None
    assert result.verdict == "UNATTRIBUTED"


def test_frame_with_no_overlap_at_all_returns_insufficient_evidence(
    density: FakeDensity,
) -> None:
    """A prior is not evidence: these vessels get no rank and no posterior."""
    tracks = [
        straight_track(219400000 + i, np.array([300000.0 + 5000.0 * i, 300000.0]), T0, 90.0, 10.0)
        for i in range(4)
    ]
    vessels: dict[int, tuple[str | None, int]] = {
        t.mmsi: ("Crude Oil Tanker", 3) for t in tracks
    }
    result = fuse_frame(tracks, vessels, density)

    assert result.candidates == []
    assert len(result.unrankable) == 4
    assert all(u.verdict == "UNATTRIBUTED" for u in result.unrankable)
    assert all(u.reason == FU.INSUFFICIENT_EVIDENCE for u in result.unrankable)
    assert result.culprit is None
    assert result.none_posterior == 1.0
    assert not any(hasattr(u, "rank") or hasattr(u, "posterior") for u in result.unrankable)


def test_no_sub_threshold_candidate_is_ever_promoted() -> None:
    """The only route from a ranking to an accusation, closed below ln(10)."""
    for log_lr in (-5.0, 0.0, 1.0, math.nextafter(FU.LOG_LR_MODERATE, 0.0)):
        scores = {
            1: C.ChannelScores(mmsi=1, s1=0.9, s2=0.9, s3=0.9, s4=1.0, t_star=T0),
            2: C.ChannelScores(mmsi=2, s1=0.1, s2=0.1, s3=0.1, s4=1.0, t_star=T0),
        }
        result = FU.FrameResult(
            candidates=FU.rank_candidates(
                {1: log_lr, 2: log_lr - 1.0},
                {1: {"s1": log_lr}, 2: {"s1": log_lr - 1.0}},
                {1: 1.0, 2: 1.0},
                scores,
            ),
            unrankable=[],
            none_posterior=0.5,
            background=(0.5, 0.5, 0.5),
        )
        assert result.candidates[0].rank == 1
        assert result.culprit is None
        assert result.verdict == "UNATTRIBUTED"


def test_unavailable_channels_reach_the_payload_with_a_reason(
    density: FakeDensity,
) -> None:
    """"E2 unavailable - no COG at t*" must be renderable, never a blank or a zero."""
    t_star_time = T0 - timedelta(minutes=30 * T_STAR_INDEX)
    origin = cloud_centre(T_STAR_INDEX)
    dark = straight_track(1, origin, t_star_time, 60.0, 12.0)
    blind = C.VesselTrack(
        mmsi=2,
        times=dark.times,
        positions_m=dark.positions_m,
        sog_kn=np.full(len(dark.times), np.nan),
        cog_deg=np.full(len(dark.times), np.nan),
    )
    scores = C.score_vessels(
        [dark, blind], density, T0, slick_orientation_deg=60.0, major_axis_km=MAJOR_AXIS_KM
    )
    assert scores[2].s2 is None
    assert scores[2].s3 is None
    assert scores[2].unavailable["s2"] == C.NO_COG_AT_T_STAR
    assert scores[2].unavailable["s3"] == C.NO_SOG_AT_T_STAR
    assert scores[1].unavailable == {}


def test_a_vessel_outside_the_drift_window_says_so(density: FakeDensity) -> None:
    origin = cloud_centre(T_STAR_INDEX)
    late = straight_track(1, origin, T0 + timedelta(days=2), 60.0, 12.0, step_min=10)
    late = C.VesselTrack(
        mmsi=1,
        times=[t + timedelta(days=2) for t in late.times],
        positions_m=late.positions_m,
        sog_kn=late.sog_kn,
        cog_deg=late.cog_deg,
    )
    scores = C.score_vessels(
        [late], density, T0, slick_orientation_deg=60.0, major_axis_km=MAJOR_AXIS_KM
    )
    assert scores[1].unavailable["s2"] == C.TRACK_DOES_NOT_SPAN_WINDOW
    assert not scores[1].has_evidence


def test_attribution_is_deterministic(
    culprit_frame: tuple[list[C.VesselTrack], dict[int, tuple[str | None, int]]],
    density: FakeDensity,
) -> None:
    tracks, vessels = culprit_frame
    first = fuse_frame(tracks, vessels, density)
    second = fuse_frame(tracks, vessels, density)
    assert [c.log_lr for c in first.candidates] == [c.log_lr for c in second.candidates]


# ------------------------------------------- §5.3 field independence (E2, E3) ----


@pytest.fixture(scope="module")
def two_ocean_states() -> tuple[FakeDensity, FakeDensity, C.VesselTrack, datetime]:
    """The same slick rewound through two genuinely different ocean states.

    Reduced scale - 500 particles over 3 h - because what is under test is which
    inputs the channels read, not the solver, which has its own suite.
    """
    from app.drift import density as D
    from app.drift import fields as F
    from app.drift import solver as S

    slick = Polygon(
        [
            (18.90, 55.300),
            (19.02, 55.340),
            (19.14, 55.352),
            (19.16, 55.336),
            (19.03, 55.318),
            (18.96, 55.286),
        ]
    )
    densities = []
    frame = None
    for seed in (7, 991):
        run = S.run(
            slick,
            "backward",
            F.synthetic_field(slick.bounds, seed=seed),
            t0=T0,
            horizon_h=3,
            n_particles=500,
            seed=42,
        )
        frame = run.frame
        densities.append(D.compute_run_density(run.states))

    assert frame is not None
    centre = slick.centroid
    origin_m = np.array(frame.to_utm.transform(float(centre.x), float(centre.y)))
    track = straight_track(1, origin_m, T0 - timedelta(minutes=90), 60.0, 12.0)
    return densities[0], densities[1], track, T0


def test_e2_and_e3_are_invariant_when_the_current_field_is_replaced(
    two_ocean_states: tuple[FakeDensity, FakeDensity, C.VesselTrack, datetime],
) -> None:
    """Swap the ocean, and E1 moves while E2 and E3 do not (§5.3).

    E1 integrates the origin probability field, so a different current field
    gives it a different answer - that is the point of the test, and the
    assertion on `mass` is what proves the two ocean states really did differ.
    E2 reads only the slick axis and the vessel's course; E3 reads only the
    observed length, the stretch factor and the speed. Neither can move, which
    is what keeps the system standing when the ocean model is coarse.
    """
    density_a, density_b, track, t0 = two_ocean_states
    kwargs = {"slick_orientation_deg": 60.0, "major_axis_km": MAJOR_AXIS_KM}
    a = C.score_vessels([track], density_a, t0, **kwargs)[1]
    b = C.score_vessels([track], density_b, t0, **kwargs)[1]

    assert a.mass != b.mass
    assert a.s2 == b.s2
    assert C.e3_kinematic_consistency(
        MAJOR_AXIS_KM, a.stretch_factor, CULPRIT_SOG_KN
    ) == C.e3_kinematic_consistency(MAJOR_AXIS_KM, a.stretch_factor, CULPRIT_SOG_KN)


def test_stretch_factor_is_the_only_field_mediated_input_to_e3(
    two_ocean_states: tuple[FakeDensity, FakeDensity, C.VesselTrack, datetime],
) -> None:
    """s3 is reproducible from the exposed stretch factor alone.

    If E3 ever grew a second, hidden route to the drift run, recomputing it from
    the published `stretch_factor` would stop matching.
    """
    density_a, density_b, track, t0 = two_ocean_states
    kwargs = {"slick_orientation_deg": 60.0, "major_axis_km": MAJOR_AXIS_KM}
    for density in (density_a, density_b):
        scored = C.score_vessels([track], density, t0, **kwargs)[1]
        assert scored.s3 == C.e3_kinematic_consistency(
            MAJOR_AXIS_KM, scored.stretch_factor, CULPRIT_SOG_KN
        )


ATTRIBUTION_DIR = Path(app.__file__).parent / "attribution"
BACKEND_DIR = Path(app.__file__).parents[1]
FORBIDDEN_PACKAGES = ("app.drift", "app.db", "app.ais")


def test_attribution_imports_nothing_from_drift_db_or_ais() -> None:
    """The §5.3 independence, enforced on the import graph rather than trusted.

    The channels take the origin probability field through a structural Protocol,
    so nothing in this package can reach a velocity field even by accident. This
    test is what stops a later refactor from quietly undoing it.
    """
    for path in sorted(ATTRIBUTION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(FORBIDDEN_PACKAGES), (
                    f"{path.name} imports {name}; attribution must not reach "
                    f"the drift field chain, the database or the AIS layer (§5.3)"
                )


def test_importing_attribution_does_not_pull_in_the_field_chain() -> None:
    """The transitive form of the same guard, checked in a clean interpreter."""
    code = (
        "import sys\n"
        "import app.attribution.channels, app.attribution.fusion, app.attribution.priors\n"
        f"leaked = sorted(m for m in sys.modules if m.startswith({FORBIDDEN_PACKAGES!r}))\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_DIR),
        check=False,
    )
    assert result.returncode == 0, result.stderr
