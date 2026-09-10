"""The fixture set and the loader that serves it — §15, §10 P0-9.

The tests that matter here are the ones about honesty and about the offline path,
not the ones about file presence:

  * The demo replays with the database stopped. Enforced on the import graph, not
    trusted: one transitive import of the data layer inside app/fallback would
    take the §2.1 offline guarantee away silently.
  * Nothing in the set claims a number it did not compute. Every null carries a
    reason, and the set says in its own manifest that it is provisional.
  * SC-02 refuses to attribute, and no code path promotes its top vessel to a
    culprit (§2.4, §5.4).
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.fallback import loader

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
FALLBACK_DIR = BACKEND_DIR / "app" / "fallback"

# app.fallback may reach none of these, transitively or otherwise: the offline
# demo has to replay with the stack down (§2.1).
FORBIDDEN_PACKAGES = ("app.db", "sqlalchemy", "asyncpg", "geoalchemy2", "redis")

SCENARIOS = ("SC-01", "SC-02", "SC-03")

CULPRIT_MMSI = 220517000


@pytest.fixture(scope="session", autouse=True)
def _fixtures_on_disk() -> None:
    """Point the loader at the repo's fixtures when the container path is absent.

    In the container settings.fixtures_dir is /data/fixtures; from a bare pytest
    at the repo root it is not mounted. Redirecting keeps `make test` and a host
    run equivalent, which is what conftest.py already does for the import path.
    """
    if not Path(settings.fixtures_dir).is_dir():
        candidate = REPO_ROOT / "data" / "fixtures"
        if not candidate.is_dir():
            pytest.skip("no fixtures on disk; run `make fixtures` first")
        settings.fixtures_dir = candidate


@pytest.fixture(scope="session")
def manifest() -> dict[str, Any]:
    return loader.manifest()


def _summary(manifest: dict[str, Any], code: str) -> dict[str, Any]:
    return next(entry for entry in manifest["scenarios"] if entry["code"] == code)


def _unexplained_nulls(
    node: Any, reasons: frozenset[str] = frozenset(), path: str = ""
) -> list[str]:
    """Null leaves with no `unavailable` entry in any enclosing object.

    Scope is inherited: a reason stated on the detection covers a null nested
    inside its features, and one stated on a WS event covers that event's
    payload. That is the shape the generator writes, and the shape the
    explainability panel reads.
    """
    found: list[str] = []
    if isinstance(node, dict):
        stated = node.get("unavailable")
        scope = reasons | frozenset(stated) if isinstance(stated, dict) else reasons
        for key, value in node.items():
            if key in ("unavailable", "feature_provenance"):
                continue
            child = f"{path}.{key}" if path else key
            if value is None:
                if key not in scope:
                    found.append(child)
            else:
                found.extend(_unexplained_nulls(value, scope, child))
    elif isinstance(node, list):
        for item in node:
            found.extend(_unexplained_nulls(item, reasons, path))
    return found


# ------------------------------------------------------------------ manifest ----


def test_the_set_declares_itself_provisional_and_unfrozen(manifest: dict[str, Any]) -> None:
    """§15's fixtures come from the real scenes; these do not, and must say so.

    A provisional set that presents itself as the finished one is exactly the
    failure §2.2 exists to prevent, and §9 only freezes a set once it is green.
    """
    assert manifest["provisional"] is True
    assert manifest["frozen"] is False
    assert manifest["why_provisional"].strip()
    assert manifest["pending"], "a provisional set must list what is still missing"


def test_manifest_badges_cover_the_simulated_inputs(manifest: dict[str, Any]) -> None:
    """Simulated data is labelled as simulated, visibly (§2.3)."""
    badges = " ".join(manifest["badges"])
    assert "SIMULATED" in badges
    assert "INJECTED" in badges


def test_tier_4_was_not_invoked(manifest: dict[str, Any]) -> None:
    """No synthetic SAR pixels anywhere: §0 Tier 4 is a human decision, not a default."""
    assert "Tier 4 was NOT invoked" in manifest["data_tier"]


def test_every_scenario_and_artefact_in_the_manifest_loads(manifest: dict[str, Any]) -> None:
    assert loader.scenarios() == list(SCENARIOS)
    for entry in manifest["scenarios"]:
        code = entry["code"]
        for artefact in entry["artefacts"]:
            assert loader.has_fixture(code, artefact), f"{code}/{artefact} missing"
            assert isinstance(loader.load(code, artefact), dict)
        assert sorted(entry["artefacts"]) == sorted(loader.artefacts(code))


# ------------------------------------------------------------------- SC-01 ----


def test_sc01_names_the_authored_culprit(manifest: dict[str, Any]) -> None:
    """The closed loop recovered the vessel that actually discharged.

    The attribution engine never saw the ground truth: it scored the observed
    slick against every vessel in the frame on equal terms.
    """
    truth = loader.load("SC-01", "truth")
    attribution = loader.load("SC-01", "attribution")

    assert truth["authored"]["culprit_mmsi"] == CULPRIT_MMSI
    assert attribution["candidates"][0]["mmsi"] == CULPRIT_MMSI
    assert truth["identified_correctly"] is True
    assert attribution["culprit_mmsi"] == CULPRIT_MMSI


def test_sc01_verdict_is_whatever_the_run_produced(manifest: dict[str, Any]) -> None:
    """The verdict is reported, never targeted.

    This asserts consistency between the manifest, the attribution and the truth
    record - not that the verdict is a particular band. §5.4's weights and §9's
    ln(10)/ln(100) thresholds are not tuned to reach one, so a change of band here
    is a finding to report, not a test to edit.
    """
    truth = loader.load("SC-01", "truth")
    attribution = loader.load("SC-01", "attribution")
    summary = _summary(manifest, "SC-01")

    assert attribution["verdict"] == truth["verdict"] == summary["verdict"]
    assert attribution["verdict"] in ("STRONG", "MODERATE", "UNATTRIBUTED")

    top = attribution["candidates"][0]
    if attribution["verdict"] == "STRONG":
        assert top["lr"] >= 100.0
    elif attribution["verdict"] == "MODERATE":
        assert 10.0 <= top["lr"] < 100.0


def test_sc01_quotes_its_frame_size_with_the_lr(manifest: dict[str, Any]) -> None:
    """Achievable LR scales with log N, so N travels with the number (§12, §16)."""
    attribution = loader.load("SC-01", "attribution")
    assert attribution["n_vessels_in_frame"] == len(loader.load("SC-01", "ais")["vessels"])
    assert attribution["n_ranked"] <= attribution["n_vessels_in_frame"]
    assert "log N" in attribution["frame_size_note"]


def test_sc01_only_the_culprit_clears_a_decision_band() -> None:
    """Every innocent vessel comes back UNATTRIBUTED. That is the product."""
    attribution = loader.load("SC-01", "attribution")
    for candidate in attribution["candidates"][1:]:
        assert candidate["verdict"] == "UNATTRIBUTED", (
            f"MMSI {candidate['mmsi']} cleared a band at LR {candidate['lr']}"
        )
        assert candidate["lr"] < 10.0


def test_sc01_recovered_the_release_point_and_time() -> None:
    """The headline claim, measured rather than asserted.

    The bound is the run's own uncertainty hull, not a hand-picked tolerance: an
    origin estimate outside the polygon the system itself draws would be the
    system contradicting its own uncertainty.
    """
    recovered = loader.load("SC-01", "truth")["recovered"]

    assert recovered["release_inside_hull"] is True
    assert math.isfinite(recovered["origin_error_km"])
    assert math.isfinite(recovered["origin_peak_error_km"])
    hull_scale_km = math.sqrt(recovered["hull_area_km2"])
    assert recovered["origin_error_km"] < hull_scale_km

    assert recovered["t_star_error_min"] is not None
    assert abs(recovered["t_star_error_min"]) <= recovered["t_star_resolution_min"]


def test_sc01_reconstructed_the_released_patch_length() -> None:
    """E3's `major_axis / stretch_factor` against the authored discharge length."""
    recovered = loader.load("SC-01", "truth")["recovered"]
    authored_km = recovered["released_length_km_authored"]

    assert recovered["released_length_km_recovered"] is not None
    assert abs(recovered["released_length_error_km"]) < 0.1 * authored_km
    # §5.1 floors the reported stretch at 1.0; the true value is recorded beside
    # it so the floor's effect on this run is visible rather than inferred.
    assert recovered["stretch_factor_at_t_star"] >= 1.0
    assert recovered["true_stretch_factor"] > 0.0


def test_sc01_e4_boosts_only_the_vessel_that_went_dark() -> None:
    """The cadence floor, visible in the fixture: clean transmitters score 1.0.

    The AIS frame is sampled at exactly settings.e4_nominal_cadence_min, so a
    vessel that never went dark must come back at exactly 1.0 (§5.3). Anything
    else means every vessel in the frame is collecting a boost for its own
    reporting interval, which inflates every log LR past a §9 threshold.
    """
    attribution = loader.load("SC-01", "attribution")
    for candidate in attribution["candidates"]:
        s4 = candidate["channels"]["s4"]
        if candidate["mmsi"] == CULPRIT_MMSI:
            assert s4 > 1.0
        else:
            assert s4 == 1.0, f"MMSI {candidate['mmsi']} got a boost for transmitting"


def test_sc01_unrankable_vessels_carry_a_reason_and_no_score() -> None:
    """A prior is not evidence: a vessel that never overlapped is not ranked (§5.4)."""
    attribution = loader.load("SC-01", "attribution")
    for entry in attribution["unrankable"]:
        assert entry["reason"]
        assert entry["verdict"] == "UNATTRIBUTED"
        assert "log_lr" not in entry and "posterior" not in entry
        assert entry["channels"]["t_star"] is None
        assert entry["channels"]["unavailable"]


def test_sc01_windage_variants_are_three_distinct_runs() -> None:
    """P1-11. Moving the slider moves a real LR, not just a redrawn polygon."""
    variants = ["drift", "drift.windage-0.020", "drift.windage-0.040"]
    runs = [loader.load("SC-01", name) for name in variants]

    assert {run["windage"] for run in runs} == {0.020, 0.033, 0.040}
    assert {run["seed"] for run in runs} == {runs[0]["seed"]}

    hulls = [run["frames"][12]["hull"]["coordinates"] for run in runs]
    assert hulls[0] != hulls[1] != hulls[2], "windage did not change the origin cloud"

    log_lrs = {
        name: loader.load("SC-01", name.replace("drift", "attribution"))["candidates"][0][
            "log_lr"
        ]
        for name in variants
    }
    assert len(set(log_lrs.values())) == 3, f"windage did not move the LR: {log_lrs}"


def test_sc01_variant_particle_subsampling_is_declared() -> None:
    """An omission is fine; an undeclared one is not (§2.2)."""
    default = loader.load("SC-01", "drift")
    variant = loader.load("SC-01", "drift.windage-0.020")

    assert "particles_subsampled_from" not in default
    assert default["frames"][0]["n_particles"] == default["n_particles"]
    assert variant["particles_subsampled_from"] == variant["n_particles"]
    assert variant["frames"][0]["n_particles"] < variant["n_particles"]


# --------------------------------------------------------- SC-02, SC-03 ----


def test_sc02_refuses_to_attribute(manifest: dict[str, Any]) -> None:
    """P0-CRITICAL. It ships, or the demo does not run (§12).

    The refusal is driven by the §5.2 wind gate evaluated against config, and it
    is a first-class output, not an error path (§2.4).
    """
    detection = loader.load("SC-02", "detection")
    attribution = detection["attribution"]

    assert detection["wind_gate_violated"] is True
    assert detection["features"]["wind_speed_ms"] < settings.wind_gate_min_ms
    assert attribution["issued"] is False
    assert attribution["verdict"] == "UNATTRIBUTED"
    assert attribution["candidates"] == []
    assert attribution["recommendation"].strip()
    assert not loader.has_fixture("SC-02", "attribution")
    assert not loader.has_fixture("SC-02", "drift")


def test_sc02_classifies_look_alike_without_claiming_a_model() -> None:
    """§12's LOOK-ALIKE, from the rule-based §5.2 scorer, labelled as such.

    There is still no trained discriminator in this build. What changed is that
    the fallback is now a real deterministic scorer over the §5.2 physics rather
    than a shrug — so there is a number, a class and a breakdown, and every one
    of them is marked `rule_based`. The number is whatever the physics produced;
    §12 was amended to match the engine rather than the weights tuned to match
    §12 (§2.2).
    """
    detection = loader.load("SC-02", "detection")

    assert detection["class"] == "look-alike"
    assert 0.0 < detection["p_oil"] < settings.discriminator_oil_threshold
    assert detection["shap_factors"], "a verdict with no breakdown is not explainable"

    # The qualifier, in every place a reader could form an impression from.
    assert detection["method"] == "rule_based"
    assert detection["detector"] == "rule_based"
    assert detection["factor_basis"] == "rule_based"
    assert "not from a trained model" in detection["note"]
    assert all(factor["basis"] == "rule_based" for factor in detection["shap_factors"])

    # A rule-based run never claims a version, and says why it has none.
    assert detection["model_version"] is None
    assert "rule-based" in detection["unavailable"]["model_version"]

    # The wind gate dominates, and it argues look-alike.
    assert detection["shap_factors"][0]["feature"] == "wind_speed_ms"
    assert detection["shap_factors"][0]["direction"] == "look-alike"


def test_sc02_contributions_sum_to_the_probability_shown() -> None:
    """The panel's bars must sum to the number above them, or it is decoration."""
    detection = loader.load("SC-02", "detection")

    base = detection["base_p_oil"]
    total = math.log(base / (1.0 - base)) + sum(
        factor["contribution"] for factor in detection["shap_factors"]
    )
    p_oil = 1.0 / (1.0 + math.exp(-total))
    assert p_oil == pytest.approx(detection["p_oil"], abs=5e-5)


def test_sc02_states_what_it_could_not_measure() -> None:
    """Two of the six rule terms need pixels there are none of. Absent, with a
    reason — never a zero, which would read as a measurement of no damping."""
    detection = loader.load("SC-02", "detection")

    for name in ("edge_gradient_mean", "contrast_db"):
        assert detection["features"][name] is None
        assert "pixels" in detection["unavailable"][name]
    # Less evidence has to mean a weaker claim, and the fixture records how much.
    assert 0.0 < detection["evidence_fraction"] < 1.0


def test_sc03_finds_nothing(manifest: dict[str, Any]) -> None:
    """Zero detections. Proves no false positives (§12)."""
    scene = loader.load("SC-03", "scene")

    assert "verdict" not in _summary(manifest, "SC-03")
    assert scene["n_detections"] == 0
    assert scene["detections"] == []
    assert settings.wind_gate_min_ms <= scene["wind_speed_ms"] <= settings.wind_gate_max_ms
    for artefact in ("detection", "drift", "attribution", "truth"):
        assert not loader.has_fixture("SC-03", artefact)


# ------------------------------------------------------------- honesty ----


@pytest.mark.parametrize("code", SCENARIOS)
def test_every_null_carries_a_reason(code: str) -> None:
    """No silent nulls anywhere in the set.

    A field that could not be computed says so and says why; the alternative is a
    blank the UI renders as a zero.
    """
    for artefact in loader.artefacts(code):
        unexplained = _unexplained_nulls(loader.load(code, artefact))
        assert not unexplained, f"{code}/{artefact}: null with no stated reason: {unexplained}"


@pytest.mark.parametrize("code", SCENARIOS)
def test_every_ais_track_is_badged_as_simulated(code: str) -> None:
    """§2.3. Every track here is injected, not only the culprit's — and says so."""
    if not loader.has_fixture(code, "ais"):
        return
    ais = loader.load(code, "ais")
    assert ais["badge"]
    for vessel in ais["vessels"]:
        assert vessel["is_injected"] is True
        assert vessel["badge"] == ais["badge"]
        assert all(position["is_injected"] for position in vessel["positions"])


@pytest.mark.parametrize("code", SCENARIOS)
def test_scene_geometry_carries_its_authored_badge(code: str) -> None:
    scene = loader.load(code, "scene")
    assert scene["badges"]
    assert scene["raster_path"] is None
    assert scene["source"] == "authored"


def test_drift_density_is_run_normalised_and_declares_its_coarsening() -> None:
    """§5.1 normalises O(x, y, t) across the whole run, not per snapshot."""
    for name in ("drift", "drift.windage-0.020", "drift.windage-0.040"):
        run = loader.load("SC-01", name)
        grid = run["density_grid"]

        assert grid["cell_size_m"] == grid["engine_cell_size_m"] * grid["coarsen_factor"]
        assert grid["crs"].startswith("EPSG:")
        retained = grid["mass_retained_run_total"]
        assert grid["mass_kept_fraction"] <= retained <= 1.0

        summed = sum(cell[2] for frame in run["frames"] for cell in frame["density_cells"])
        assert math.isclose(summed, retained, rel_tol=1e-3)


def test_ws_events_match_the_schema_and_separate_timing_from_pacing() -> None:
    """The §7 shape exactly — a rename here breaks the console mid-demo.

    `playback_offset_ms` is authored demo pacing and `timings_ms` is measured
    engine time. They are separate fields so the UI can never render one as the
    other (§2.2).
    """
    from app.schemas.ws import PipelineEvent

    for code in SCENARIOS:
        payload = loader.load(code, "ws")
        events = loader.load_ws_sequence(code)

        assert events, f"{code} has no WS sequence"
        assert payload["timings_note"].strip()

        offsets = [event["playback_offset_ms"] for event in events]
        assert offsets == sorted(offsets)
        assert payload["total_playback_ms"] == offsets[-1]

        for event in events:
            PipelineEvent.model_validate(
                {k: v for k, v in event.items() if k != "playback_offset_ms"}
            )
        assert events[-1]["stage"] == "DONE"
        assert events[-1]["progress"] == 1.0


def test_ws_stage_messages_are_written_for_a_human() -> None:
    """§7: stage messages are what make a judge watch the system think."""
    for event in loader.load_ws_sequence("SC-01"):
        assert event["message"].strip()
        assert "Processing" not in event["message"]
    rewinds = [e for e in loader.load_ws_sequence("SC-01") if e["stage"] == "REWINDING"]
    assert any("T−" in event["message"] for event in rewinds)


# -------------------------------------------------------------- loader ----


def test_loader_stamps_the_fixture_source() -> None:
    """§7. A fixture-backed answer is never presented as a live one."""
    enveloped = loader.envelope(loader.load("SC-01", "scene"), elapsed_ms=12)
    assert enveloped["source"] == "fixture"
    assert enveloped["elapsed_ms"] == 12
    assert enveloped["id"] == "SC-01"


def test_loader_rejects_names_that_are_not_fixture_keys() -> None:
    for bad in ("../../etc/passwd", "..", "SC-01/../../x"):
        assert loader.has_fixture("SC-01", bad) is False
        with pytest.raises(loader.MalformedFixtureNameError):
            loader.load("SC-01", bad)
    with pytest.raises(loader.MalformedFixtureNameError):
        loader.load("../SC-01", "scene")


def test_loader_raises_a_typed_error_for_a_missing_fixture() -> None:
    assert loader.has_fixture("SC-03", "attribution") is False
    with pytest.raises(loader.FixtureNotFoundError):
        loader.load("SC-03", "attribution")


def test_loader_never_writes() -> None:
    """The set is FROZEN once green (§9); this module reads and nothing else."""
    source = (FALLBACK_DIR / "loader.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (
            "write_text",
            "write_bytes",
            "mkdir",
            "unlink",
            "rmdir",
        ):
            pytest.fail(f"loader.py calls {node.attr}; fixtures are read-only (§9)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                pytest.fail("loader.py calls bare open(); use Path.open in read mode")


# --------------------------------------------- the offline guarantee ----


def test_fallback_package_imports_no_data_layer() -> None:
    """§2.1. Checked on the source, so a refactor cannot quietly undo it."""
    for path in sorted(FALLBACK_DIR.glob("*.py")):
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
                    f"{path.name} imports {name}; the fixture path must replay "
                    f"with the database stopped (§2.1)"
                )


def test_importing_the_loader_pulls_in_no_data_layer() -> None:
    """The transitive form, in a clean interpreter — the demo runs with db down."""
    code = (
        "import sys\n"
        "import app.fallback.loader\n"
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


def test_the_whole_set_replays_from_disk_alone() -> None:
    """Every artefact parses without a database, a network or a model file."""
    total = 0
    for code in loader.scenarios():
        for artefact in loader.artefacts(code):
            payload = loader.load(code, artefact)
            total += len(json.dumps(payload))
    assert total > 0
    assert loader.manifest()["scenarios"]
