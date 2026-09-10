"""Drift solver, field chain and density — §5.1.

The physics tests here are the ones that matter: reversibility under a steady
field is what proves the RK4 integrator and the backward mode are right, and the
non-divergence of the synthetic current is what makes it geostrophic-*like*
rather than decorative noise. Determinism is a §2.1 requirement, not a nicety -
the demo runs offline and must reproduce exactly.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from shapely.geometry import Polygon

from app.config import settings
from app.drift import density as D
from app.drift import fields as F
from app.drift import solver as S

T0 = datetime(2024, 3, 1, 2, 0, tzinfo=UTC)

# A plausible Baltic slick: elongated, concave, well away from a UTM zone edge.
SLICK = Polygon(
    [
        (18.90, 55.300),
        (19.02, 55.340),
        (19.14, 55.352),
        (19.16, 55.336),
        (19.03, 55.318),
        (18.96, 55.286),
    ]
)


@pytest.fixture(scope="module")
def field() -> F.SyntheticField:
    return F.synthetic_field(SLICK.bounds, seed=7)


@pytest.fixture(scope="module")
def full_run(field: F.SyntheticField) -> S.DriftRun:
    """One full-scale 5000-particle 12 h backward run, shared across tests."""
    return S.run(SLICK, "backward", field, t0=T0, seed=42)


# ------------------------------------------------------------- field chain ----


def test_chain_reaches_synthetic_with_no_netcdf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "fixtures_dir", tmp_path / "fixtures")
    resolved = F.resolve(SLICK.bounds, T0, scenario_code=None, seed=7)
    assert resolved.source == "synthetic"


def test_chain_reports_scenario_snapshot_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "fixtures_dir", tmp_path)
    fields_dir = tmp_path / "fields"
    fields_dir.mkdir()
    lons = np.linspace(18.8, 19.3, 8)
    lats = np.linspace(55.2, 55.5, 6)
    zeros = np.zeros((lats.size, lons.size))
    np.savez(
        fields_dir / "SC-01.npz",
        lons=lons,
        lats=lats,
        u_current=zeros + 0.1,
        v_current=zeros,
        u_wind=zeros + 5.0,
        v_wind=zeros,
    )
    resolved = F.resolve(SLICK.bounds, T0, scenario_code="SC-01", seed=7)
    assert resolved.source == "scenario_snapshot"


def test_unreadable_snapshot_degrades_rather_than_raising(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "fixtures_dir", tmp_path)
    fields_dir = tmp_path / "fields"
    fields_dir.mkdir()
    (fields_dir / "SC-01.npz").write_bytes(b"not an npz")
    resolved = F.resolve(SLICK.bounds, T0, scenario_code="SC-01", seed=7)
    assert resolved.source == "synthetic"


def test_synthetic_field_is_deterministic_given_seed() -> None:
    lon = np.linspace(18.9, 19.1, 32)
    lat = np.full_like(lon, 55.32)
    a = F.synthetic_field(SLICK.bounds, seed=11).current_ms(lon, lat, T0)
    b = F.synthetic_field(SLICK.bounds, seed=11).current_ms(lon, lat, T0)
    c = F.synthetic_field(SLICK.bounds, seed=12).current_ms(lon, lat, T0)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_synthetic_current_is_non_divergent(field: F.SyntheticField) -> None:
    """div u ~ 0. A divergent synthetic current would pile particles into
    convergence zones no ocean produced and hand the KDE a false concentration."""
    x = np.linspace(-2.0e4, 2.0e4, 40)
    grid_x, grid_y = np.meshgrid(x, x)
    lon = field.lon0 + np.degrees(grid_x / (6_371_000.0 * np.cos(np.radians(field.lat0))))
    lat = field.lat0 + np.degrees(grid_y / 6_371_000.0)

    vel = field.current_ms(lon.ravel(), lat.ravel(), T0).reshape(*grid_x.shape, 2)
    step = x[1] - x[0]
    du_dx = np.gradient(vel[..., 0], step, axis=1)
    dv_dy = np.gradient(vel[..., 1], step, axis=0)

    # Interior only: np.gradient falls back to one-sided differences on the edge
    # rows, whose O(h) truncation error swamps the O(h^2) interior and would make
    # this a test of numpy's edge handling rather than of the field.
    divergence = (du_dx + dv_dy)[1:-1, 1:-1]
    shear_scale = max(np.abs(du_dx).max(), np.abs(dv_dy).max())
    assert np.abs(divergence).max() < 0.01 * shear_scale


def test_synthetic_wind_is_spatially_coherent(field: F.SyntheticField) -> None:
    lon = np.linspace(18.9, 19.1, 64)
    lat = np.full_like(lon, 55.32)
    wind = field.wind10_ms(lon, lat, T0)
    direction = np.arctan2(wind[:, 1], wind[:, 0])
    assert np.ptp(np.unwrap(direction)) < np.radians(45.0)


def test_stokes_is_a_fraction_of_wind(field: F.SyntheticField) -> None:
    lon = np.linspace(18.9, 19.1, 16)
    lat = np.full_like(lon, 55.32)
    wind = field.wind10_ms(lon, lat, T0)
    stokes = field.stokes_ms(lon, lat, T0)
    assert np.allclose(stokes, settings.stokes_wind_fraction * wind)


# ------------------------------------------------------- particle seeding ----


def test_seeded_particles_are_inside_the_polygon() -> None:
    pts = S.seed_particles(SLICK, 2000, seed=3)
    assert pts.shape == (2000, 2)
    import shapely

    assert bool(shapely.contains_xy(SLICK, pts[:, 0], pts[:, 1]).all())


def test_seeding_is_area_weighted() -> None:
    """Two disjoint lobes, 3:1 in area, sampled in proportion."""
    big = Polygon([(0.0, 0.0), (0.3, 0.0), (0.3, 0.1), (0.0, 0.1)])
    small = Polygon([(0.4, 0.0), (0.5, 0.0), (0.5, 0.1), (0.4, 0.1)])
    both = big.union(small)
    pts = S.seed_particles(both, 8000, seed=5)
    in_big = (pts[:, 0] < 0.35).sum() / pts.shape[0]
    assert in_big == pytest.approx(0.75, abs=0.02)


def test_seeding_is_deterministic() -> None:
    assert np.array_equal(S.seed_particles(SLICK, 500, 9), S.seed_particles(SLICK, 500, 9))


def test_degenerate_polygon_raises() -> None:
    with pytest.raises(S.DegenerateSlickError):
        S.seed_particles(Polygon(), 100, seed=1)


# ------------------------------------------------------------- integration ----


def test_backward_run_emits_25_snapshots_over_12h(full_run: S.DriftRun) -> None:
    assert len(full_run.states) == 25
    assert [s.t_offset_min for s in full_run.states] == list(range(0, -750, -30))
    assert full_run.states[-1].t_offset_min == -720
    assert full_run.n_particles == 5000


def test_forward_run_emits_positive_offsets(field: F.SyntheticField) -> None:
    run = S.run(SLICK, "forward", field, t0=T0, n_particles=400, seed=42)
    assert [s.t_offset_min for s in run.states] == list(range(0, 750, 30))


def test_snapshot_time_is_t0_plus_signed_offset(full_run: S.DriftRun) -> None:
    assert S.snapshot_time(full_run, 0) == T0
    assert (T0 - S.snapshot_time(full_run, -1)).total_seconds() == 12 * 3600


def test_field_source_is_reported(full_run: S.DriftRun) -> None:
    assert full_run.field_source == "synthetic"


def test_run_is_bit_identical_for_the_same_seed(field: F.SyntheticField) -> None:
    a = S.run(SLICK, "backward", field, t0=T0, n_particles=600, seed=42)
    b = S.run(SLICK, "backward", field, t0=T0, n_particles=600, seed=42)
    for sa, sb in zip(a.states, b.states, strict=True):
        assert np.array_equal(sa.positions, sb.positions)
        assert sa.stretch_factor == sb.stretch_factor


def test_different_seed_gives_different_output(field: F.SyntheticField) -> None:
    a = S.run(SLICK, "backward", field, t0=T0, n_particles=600, seed=42)
    b = S.run(SLICK, "backward", field, t0=T0, n_particles=600, seed=43)
    assert not np.array_equal(a.states[-1].positions, b.states[-1].positions)


def test_backward_then_forward_closes(field: F.SyntheticField) -> None:
    """With diffusion off, rewinding and replaying must return to the start.

    This is the real accuracy proof for both the RK4 step and the backward mode:
    an integrator that flipped the velocity sign but kept advancing model time
    forward would not close.
    """
    back = S.run(SLICK, "backward", field, t0=T0, n_particles=300, k_h=0.0, seed=1)
    end = back.states[-1].positions
    lon, lat = back.frame.inverse(end)

    positions = end.copy()
    t_end_s = (T0.timestamp()) - 12 * 3600
    for step in range(int(12 * 3600 / settings.drift_dt_s)):
        positions = S.rk4_step(
            positions,
            t_end_s + step * settings.drift_dt_s,
            settings.drift_dt_s,
            field,
            1,
            settings.windage_alpha,
            settings.theta_dev_deg,
            back.frame,
        )
    del lon, lat
    drift_m = np.linalg.norm(positions - back.states[0].positions, axis=1)
    assert float(drift_m.max()) < 1.0


def test_diffusion_matches_the_specified_sigma() -> None:
    rng = np.random.default_rng(0)
    positions = np.zeros((200_000, 2))
    moved = S.diffusion_step(positions, k_h=10.0, dt_s=300.0, rng=rng)
    expected_sigma = np.sqrt(2.0 * 10.0 * 300.0)
    assert float(moved.std()) == pytest.approx(expected_sigma, rel=0.02)


def test_diffusion_applies_in_both_modes(field: F.SyntheticField) -> None:
    """Symmetric under time reversal. A reverse run without it would report a
    false-precision origin (§2.2)."""
    spread = {}
    for mode in ("forward", "backward"):
        without = S.run(SLICK, mode, field, t0=T0, n_particles=400, k_h=0.0, seed=2)
        with_diff = S.run(SLICK, mode, field, t0=T0, n_particles=400, k_h=10.0, seed=2)
        spread[mode] = (
            S.major_axis_m(with_diff.states[-1].positions),
            S.major_axis_m(without.states[-1].positions),
        )
    for mode, (diffused, clean) in spread.items():
        assert diffused > clean, f"{mode} run did not diffuse"


def test_windage_outside_the_spec_range_raises(field: F.SyntheticField) -> None:
    with pytest.raises(ValueError, match="windage"):
        S.run(SLICK, "backward", field, t0=T0, n_particles=50, windage=0.10)


def test_theta_dev_outside_the_spec_range_raises(field: F.SyntheticField) -> None:
    with pytest.raises(ValueError, match="theta_dev_deg"):
        S.run(SLICK, "backward", field, t0=T0, n_particles=50, theta_dev_deg=45.0)


def test_snapshot_interval_must_divide_into_steps(field: F.SyntheticField) -> None:
    with pytest.raises(ValueError, match="whole number"):
        S.run(SLICK, "backward", field, t0=T0, n_particles=50, dt_s=700)


def test_n_particles_and_horizon_are_run_parameters(field: F.SyntheticField) -> None:
    """The what-if sliders and a responsiveness-constrained demo override these
    per run; they must not be wired to the config defaults."""
    run = S.run(SLICK, "backward", field, t0=T0, n_particles=250, horizon_h=6, seed=1)
    assert run.n_particles == 250
    assert run.horizon_h == 6
    assert run.states[0].positions.shape[0] == 250
    assert run.states[-1].t_offset_min == -360
    assert len(run.states) == 13


def test_theta_dev_rotates_the_windage_term(field: F.SyntheticField) -> None:
    frame = S.Frame.for_point(19.0, 55.32)
    positions = frame.forward(np.array([19.00, 19.02]), np.array([55.32, 55.33]))
    straight = S.velocity(positions, T0.timestamp(), field, 0.033, 0.0, frame)
    veered = S.velocity(positions, T0.timestamp(), field, 0.033, 20.0, frame)
    assert not np.allclose(straight, veered)


# ---------------------------------------------------------- stretch factor ----


def test_stretch_factor_is_one_at_the_seed(full_run: S.DriftRun) -> None:
    assert full_run.states[0].stretch_factor == 1.0


def test_stretch_factor_is_never_below_one(full_run: S.DriftRun) -> None:
    """Oil spreads; it does not contract. A released patch longer than the
    observed slick is physically impossible and E3 divides by this (§5.3)."""
    assert all(s.stretch_factor >= 1.0 for s in full_run.states)


def test_raw_stretch_factor_is_reported_unclamped(full_run: S.DriftRun) -> None:
    """The floor is a stated bound, not a silent correction - the uncorrected
    value stays visible (§2.2)."""
    assert any(s.stretch_factor_raw < s.stretch_factor for s in full_run.states)


def test_backward_stretch_inverts_the_forward_ratio() -> None:
    seed_cloud = np.column_stack([np.linspace(-5000, 5000, 400), np.zeros(400)])
    stretched = seed_cloud * np.array([2.0, 1.0])
    back_reported, back_raw = S.stretch_factor(stretched, seed_cloud, "backward")
    fwd_reported, fwd_raw = S.stretch_factor(stretched, seed_cloud, "forward")
    assert back_raw == pytest.approx(0.5, rel=1e-6)
    assert fwd_raw == pytest.approx(2.0, rel=1e-6)
    assert back_reported == 1.0
    assert fwd_reported == pytest.approx(2.0, rel=1e-6)


def test_stretch_factor_removes_diffusive_variance() -> None:
    """Diffusion is isotropic uncertainty, not strain. Left in, it would make a
    reverse run's 'stretch' mostly a measure of its own K_h."""
    rng = np.random.default_rng(0)
    seed_cloud = np.column_stack([np.linspace(-5000, 5000, 4000), np.zeros(4000)])
    elapsed_s = 43200.0
    k_h = 10.0
    diffused = seed_cloud + rng.normal(0.0, np.sqrt(2.0 * k_h * elapsed_s), size=seed_cloud.shape)

    uncorrected, _ = S.stretch_factor(diffused, seed_cloud, "forward")
    corrected, _ = S.stretch_factor(diffused, seed_cloud, "forward", k_h=k_h, elapsed_s=elapsed_s)
    assert corrected < uncorrected
    assert corrected == pytest.approx(1.0, abs=0.02)


def test_degenerate_seed_cloud_clamps_to_one() -> None:
    point = np.zeros((10, 2))
    reported, raw = S.stretch_factor(np.ones((10, 2)), point, "backward")
    assert reported == 1.0
    assert raw == 1.0


# ------------------------------------------------------------------ density ----


def test_density_normalises_across_the_whole_run(full_run: S.DriftRun) -> None:
    """Sum over every cell of every snapshot is 1.0 — E1 integrates mass over
    time, so per-snapshot normalisation would flatten the time dimension."""
    result = D.compute_run_density(full_run.states)
    assert result.total_mass() == pytest.approx(1.0, rel=1e-9)
    assert len(result.snapshots) == 25


def test_no_single_snapshot_is_normalised_alone(full_run: S.DriftRun) -> None:
    result = D.compute_run_density(full_run.states)
    per_snapshot = [float(s.probability.sum()) for s in result.snapshots]
    assert all(m < 1.0 for m in per_snapshot)
    assert max(per_snapshot) > min(per_snapshot)


def test_density_grid_is_shared_by_every_snapshot(full_run: S.DriftRun) -> None:
    result = D.compute_run_density(full_run.states)
    shapes = {s.probability.shape for s in result.snapshots}
    assert shapes == {result.grid.shape}


def test_normalise_run_rejects_an_empty_run() -> None:
    with pytest.raises(ValueError, match="zero"):
        D.normalise_run([np.zeros((4, 4))])


def test_hulls_are_valid_and_contain_most_particles(full_run: S.DriftRun) -> None:
    import shapely

    result = D.compute_run_density(full_run.states)
    for state, snapshot in zip(full_run.states, result.snapshots, strict=True):
        assert snapshot.hull.is_valid and not snapshot.hull.is_empty
        # `covers`, not `contains`: the hull's own vertices are particles and lie
        # exactly on its boundary, which `contains` excludes by definition.
        inside = shapely.covers(snapshot.hull, shapely.points(state.positions)).mean()
        assert inside > 0.95


def test_alpha_hull_follows_a_concavity() -> None:
    """A convex hull would bridge the notch and claim origin probability where
    no particle ever went."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5000, 5000, size=(6000, 2))
    keep = ~((np.abs(pts[:, 0]) < 1500) & (pts[:, 1] > 0))
    cloud = pts[keep]
    hull = D.alpha_hull(cloud)
    assert hull.area < 0.92 * hull.convex_hull.area


def test_alpha_hull_degrades_to_convex_for_a_tiny_cloud() -> None:
    hull = D.alpha_hull(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert not hull.is_empty


def test_density_is_deterministic(field: F.SyntheticField) -> None:
    run_a = S.run(SLICK, "backward", field, t0=T0, n_particles=500, seed=8)
    run_b = S.run(SLICK, "backward", field, t0=T0, n_particles=500, seed=8)
    a = D.compute_run_density(run_a.states)
    b = D.compute_run_density(run_b.states)
    for sa, sb in zip(a.snapshots, b.snapshots, strict=True):
        assert np.array_equal(sa.probability, sb.probability)
        assert sa.hull.equals(sb.hull)


# ---------------------------------------------------------------- budget ----


def test_full_run_with_density_fits_the_live_stage_budget(field: F.SyntheticField) -> None:
    """§7 streams the pipeline live. A 5000-particle 12 h rewind that takes tens
    of seconds makes the WS stage sequence unwatchable, so the budget is asserted
    here rather than discovered on stage.
    """
    import time

    started = time.perf_counter()
    run = S.run(SLICK, "backward", field, t0=T0, seed=42)
    D.compute_run_density(run.states)
    elapsed_s = time.perf_counter() - started

    assert run.n_particles == 5000
    assert len(run.states) == 25
    assert elapsed_s < settings.drift_budget_s, (
        f"full backward run + density took {elapsed_s:.2f} s, "
        f"budget is {settings.drift_budget_s} s"
    )
