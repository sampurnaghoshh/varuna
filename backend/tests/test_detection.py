"""Dark-formation detection, features, discrimination and explanation — §5.2.

Detection is stage 1 of 4 and the least interesting part of the system (§1), so
most of what is tested here is not detection quality. It is the honesty of the
output:

  * `classical.py` never imports torch, so the fallback survives a laptop that
    cannot load the U-Net (§9). Enforced on the import graph, not trusted.
  * A rule-based score is labelled `rule_based` everywhere it surfaces — result,
    UI payload and persisted row — and never claims a `model_version`. A
    rule-based scorer is defensible; one presented as a trained model is not.
  * An unmeasurable feature is absent with a reason, never a zero. A zero
    contrast reads as a measurement of no damping (§2.2).
  * With no evidence at all there is no score. `p_oil = None` is a first-class
    outcome, the same way `UNATTRIBUTED` is (§2.4).
  * SC-02 returns LOOK-ALIKE and SC-03 returns nothing at all (§12).
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import app
from app.config import settings
from app.detection import classical, explain
from app.detection import discriminator as D
from app.detection import features as F

DETECTION_DIR = Path(app.__file__).parent / "detection"
BACKEND_DIR = Path(app.__file__).parents[1]

# --- a synthetic scene --------------------------------------------------------
# 600 x 600 at 20 m: big enough for the 201 px background window to have room
# around a slick, small enough to stay fast in the suite.
SCENE_PX = 600
PIXEL_M = 20.0
SEA_DB = -12.0
TRANSFORM = (PIXEL_M, 0.0, 500_000.0, 0.0, -PIXEL_M, 6_100_000.0)


def _sea(seed: int = 42) -> np.ndarray:
    """Speckled open water with no dark formation in it."""
    rng = np.random.default_rng(seed)
    return SEA_DB + rng.normal(0.0, 0.8, (SCENE_PX, SCENE_PX))


def _with_slick(
    sea: np.ndarray, semi_major_px: float = 120.0, semi_minor_px: float = 35.0, depth_db: float = 6.0
) -> np.ndarray:
    """Plant an elongated ellipse `depth_db` below the surrounding sea."""
    rows, cols = np.mgrid[0:SCENE_PX, 0:SCENE_PX]
    angle = np.deg2rad(35.0)
    centre = SCENE_PX / 2.0
    x = (cols - centre) * np.cos(angle) + (rows - centre) * np.sin(angle)
    y = -(cols - centre) * np.sin(angle) + (rows - centre) * np.cos(angle)
    inside = (x / semi_major_px) ** 2 + (y / semi_minor_px) ** 2 <= 1.0
    return sea - depth_db * inside


def _features(**overrides: object) -> F.ExtractedFeatures:
    """A feature vector carrying only what is named, everything else absent."""
    values = dict.fromkeys(F.FEATURE_NAMES)
    values.update(overrides)
    unavailable = {name: "not measured in this test" for name, v in values.items() if v is None}
    return F.ExtractedFeatures(values=values, unavailable=unavailable)  # type: ignore[arg-type]


def _sc02_features() -> F.ExtractedFeatures:
    """SC-02's real feature values, read from the fixture the demo replays."""
    path = Path(settings.fixtures_dir) / "SC-02" / "detection.json"
    detection = json.loads(path.read_text(encoding="utf-8"))
    values = dict.fromkeys(F.FEATURE_NAMES)
    values.update(detection["features"])
    unavailable = {
        name: reason
        for name, reason in detection["unavailable"].items()
        if name in F.FEATURE_NAMES
    }
    return F.ExtractedFeatures(values=values, unavailable=unavailable)  # type: ignore[arg-type]


# ---------------------------------------------------- §9: the fallback stays ----


def test_classical_never_imports_torch() -> None:
    """§9's rule, enforced on the import graph rather than trusted.

    `classical.py` is why the demo survives a missing weights file or a broken
    CUDA install. One convenience import of a heavyweight stack takes that away
    silently — the module still works on the dev machine, and fails on the one
    that matters.
    """
    tree = ast.parse((DETECTION_DIR / "classical.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    for name in imported:
        assert not name.startswith(("torch", "segmentation_models_pytorch")), (
            f"classical.py imports {name}; it is the always-works fallback and "
            f"must stay dependency-light (§9)"
        )
    # rasterio and skimage are pinned and available, but importing either here
    # would tie the fallback to a GDAL stack it exists to be independent of.
    for name in imported:
        assert not name.startswith(("rasterio", "skimage")), (
            f"classical.py imports {name}; polygonisation is hand-rolled on "
            f"purpose to keep this module at numpy + scipy + shapely (§9)"
        )


def test_importing_classical_does_not_pull_in_torch() -> None:
    """The transitive form of the same guard, in a clean interpreter."""
    code = (
        "import sys\n"
        "import app.detection.classical\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] "
        "in ('torch', 'rasterio', 'skimage', 'lightgbm', 'shap'))\n"
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


# ------------------------------------------------------------------- CFAR ----


def test_clean_sea_yields_no_detections() -> None:
    """SC-03 'Clean Sea' — zero detections proves no false positives (§12)."""
    assert classical.detect(_sea(), TRANSFORM, settings.detection_min_area_km2) == []


def test_cfar_recovers_a_planted_slick() -> None:
    """Area and shape come back within a few percent of the planted truth."""
    polygons = classical.detect(_with_slick(_sea()), TRANSFORM, settings.detection_min_area_km2)
    assert len(polygons) == 1

    geometry = F.geometry_features(polygons[0])
    truth_km2 = math.pi * 120.0 * 35.0 * PIXEL_M**2 / 1.0e6
    assert geometry["area_km2"] == pytest.approx(truth_km2, rel=0.05)
    # An ellipse of these semi-axes has eccentricity sqrt(1 - (35/120)^2).
    assert geometry["eccentricity"] == pytest.approx(math.sqrt(1.0 - (35.0 / 120.0) ** 2), abs=0.05)


def test_area_floor_drops_formations_below_it() -> None:
    """The floor exists so the §5.1 solver does not seed 5000 particles in noise."""
    scene = _with_slick(_sea())
    assert classical.detect(scene, TRANSFORM, 20.0) == []
    assert len(classical.detect(scene, TRANSFORM, 0.5)) == 1


def test_a_geographic_transform_is_refused_not_measured_in_degrees() -> None:
    """§8: never compute distances in degrees. Loud, because no scale repairs it."""
    degrees = (0.0002, 0.0, 14.6, 0.0, -0.0002, 55.5)
    with pytest.raises(ValueError, match="degrees"):
        classical.detect(_sea(), degrees, 0.5)


def test_flat_water_is_suppressed_rather_than_thresholded() -> None:
    """A window with no contrast carries no information to threshold.

    Without the local-std floor a perfectly flat patch produces detections at
    whatever rate the window is wide — pure speckle promoted to evidence.
    """
    flat = np.full((SCENE_PX, SCENE_PX), SEA_DB)
    assert not classical.adaptive_threshold(flat, settings.cfar_window_px, 2.0).any()


# --------------------------------------------------------------- features ----


def test_a_circle_has_shape_complexity_one() -> None:
    """P^2/4*pi*A is 1.0 for a circle by construction — the scale's anchor."""
    circle = Polygon(
        [
            (5_000.0 * math.cos(t), 5_000.0 * math.sin(t))
            for t in np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
        ]
    )
    geometry = F.geometry_features(circle)
    assert geometry["shape_complexity"] == pytest.approx(1.0, abs=0.01)
    assert geometry["eccentricity"] == pytest.approx(0.0, abs=0.02)


def test_an_elongated_rectangle_reports_its_axes() -> None:
    box = Polygon([(0.0, 0.0), (8_000.0, 0.0), (8_000.0, 1_000.0), (0.0, 1_000.0)])
    geometry = F.geometry_features(box)
    assert geometry["major_axis_km"] == pytest.approx(8.0)
    assert geometry["minor_axis_km"] == pytest.approx(1.0)
    assert geometry["area_km2"] == pytest.approx(8.0)
    assert geometry["eccentricity"] == pytest.approx(math.sqrt(1.0 - (1.0 / 8.0) ** 2), abs=1e-6)


@pytest.mark.parametrize(
    ("wind_ms", "violated"),
    [(2.1, True), (2.99, True), (3.0, False), (7.4, False), (12.0, False), (13.0, True)],
)
def test_wind_gate_boundaries(wind_ms: float, violated: bool) -> None:
    """Below 3 m/s the sea mimics oil; above 12 m/s slicks disperse (§5.2)."""
    assert F.wind_gate_violated(wind_ms) is violated


def test_wind_gate_carries_its_reason_not_just_a_boolean() -> None:
    """The flag is one of our strongest Q&A answers, and only if it is legible."""
    assert "mimics oil" in F.wind_gate_reason(2.1)
    assert "disperse" in F.wind_gate_reason(13.0)
    assert "inside" in F.wind_gate_reason(7.4)


def test_absent_pixels_produce_reasons_not_zeros() -> None:
    """A zero contrast reads as a measurement of no damping. It is not one (§2.2)."""
    polygon = Polygon([(0.0, 0.0), (4_000.0, 0.0), (4_000.0, 2_000.0), (0.0, 2_000.0)])
    extracted = F.extract(polygon, sigma0_db=None, wind_speed_ms=2.1)

    for name in F.RADIOMETRIC_FEATURES + F.TEXTURE_FEATURES:
        assert extracted.get(name) is None
        assert extracted.unavailable[name] == F.NO_PIXELS
    assert extracted.unavailable["distance_to_coast_km"] == F.NO_COASTLINE
    assert extracted.unavailable["n_ships_within_20km"] == F.NO_AIS_FRAME
    # Geometry and wind are real regardless — this is a supported path, not a
    # degraded one.
    assert extracted.get("area_km2") == pytest.approx(8.0)
    assert extracted.get("wind_speed_ms") == 2.1


def test_features_measured_from_pixels_when_a_raster_is_present() -> None:
    """With a raster the eight image features are real and nothing is absent."""
    scene = _with_slick(_sea())
    polygon = classical.detect(scene, TRANSFORM, settings.detection_min_area_km2)[0]
    extracted = F.extract(polygon, sigma0_db=scene, transform=TRANSFORM, wind_speed_ms=7.0)

    for name in F.RADIOMETRIC_FEATURES + F.TEXTURE_FEATURES:
        assert extracted.get(name) is not None, name
        assert name not in extracted.unavailable
    # The slick was planted 6 dB down, so contrast against the annulus is
    # strongly negative and the mean sits below the surrounding sea.
    assert extracted.get("contrast_db") < -3.0
    assert extracted.get("mean_sigma0_db") < SEA_DB


def test_ship_count_is_absent_rather_than_zero_without_an_ais_frame() -> None:
    """`n_ships_within_20km = 0` is a claim about an empty sea, not a default."""
    polygon = Polygon([(0.0, 0.0), (1_000.0, 0.0), (1_000.0, 1_000.0), (0.0, 1_000.0)])
    absent, _ = F.context_features(polygon, 7.0)
    assert absent["n_ships_within_20km"] is None

    counted, unavailable = F.context_features(
        polygon, 7.0, vessel_positions_m=[(500.0, 500.0), (5_000.0, 0.0), (400_000.0, 0.0)]
    )
    assert counted["n_ships_within_20km"] == 2
    assert "n_ships_within_20km" not in unavailable


# ----------------------------------------------------------- discriminator ----


def test_no_model_means_rule_based_and_says_so() -> None:
    """The load-bearing honesty property of this whole build."""
    result = D.score(_features(wind_speed_ms=7.0, shape_complexity=3.0, eccentricity=0.9))

    assert result.method == "rule_based"
    assert result.basis == "rule_based"
    assert result.model_version is None
    assert "no trained discriminator" in result.note.lower()
    assert "not from a trained model" in result.note
    assert 0.0 < result.p_oil < 1.0


def test_a_missing_model_file_loads_as_none_rather_than_raising() -> None:
    """A missing model is the normal case in this build, not a failure (§8)."""
    assert D.load_model(Path("/data/models/does-not-exist.txt")) is None


def test_rule_based_scoring_is_deterministic() -> None:
    """Same features, same number, every time — the demo replays identically."""
    features = _features(wind_speed_ms=2.1, shape_complexity=1.7, eccentricity=0.34, area_km2=30.0)
    scores = {D.score(features).p_oil for _ in range(5)}
    assert len(scores) == 1


def test_the_scorer_separates_oil_from_look_alikes() -> None:
    """It has to discriminate, or labelling it honestly is beside the point."""
    oil = D.score(
        _features(
            wind_speed_ms=7.0,
            shape_complexity=4.0,
            eccentricity=0.95,
            area_km2=5.0,
            edge_gradient_mean=1.2,
            contrast_db=-6.0,
        )
    )
    look_alike = D.score(
        _features(wind_speed_ms=2.1, shape_complexity=1.4, eccentricity=0.2, area_km2=400.0)
    )
    assert oil.p_class == D.CLASS_OIL
    assert look_alike.p_class == D.CLASS_LOOK_ALIKE
    assert oil.p_oil > 0.8
    assert look_alike.p_oil < 0.2


def test_missing_evidence_costs_confidence_in_both_directions() -> None:
    """Fewer terms must mean a weaker claim, not the same claim asserted anyway."""
    full = dict(
        wind_speed_ms=7.0,
        shape_complexity=4.0,
        eccentricity=0.95,
        area_km2=5.0,
        edge_gradient_mean=1.2,
        contrast_db=-6.0,
    )
    with_pixels = D.score(_features(**full))
    without_pixels = D.score(
        _features(**{k: v for k, v in full.items() if k not in ("edge_gradient_mean", "contrast_db")})
    )

    assert without_pixels.evidence_fraction < with_pixels.evidence_fraction == 1.0
    # Both say oil; the one that measured less says it less confidently.
    assert without_pixels.p_oil < with_pixels.p_oil
    assert without_pixels.p_oil > with_pixels.base_p_oil


def test_no_score_comes_out_of_no_evidence() -> None:
    """A logistic over an empty term set returns the base rate, and a base rate
    on screen is indistinguishable from a measurement. So there is no number
    (§2.2) — the same refusal `UNATTRIBUTED` makes downstream (§2.4)."""
    result = D.score(_features())

    assert result.p_oil is None
    assert result.p_class is None
    assert result.factors == []
    assert "insufficient features" in result.unavailable["p_oil"]
    assert "insufficient features" in result.unavailable["class"]
    assert result.method == "rule_based"
    assert result.note


def test_one_surviving_term_is_enough_to_score() -> None:
    """The refusal is for no evidence, not for incomplete evidence."""
    assert D.score(_features(wind_speed_ms=2.1)).p_oil is not None


def test_classify_never_returns_oil_free_for_a_detection() -> None:
    """A detection exists, so it is oil or a look-alike. 'oil-free' is what a
    scene with nothing in it is, and it is reached through `scene_class`."""
    assert D.classify(0.9) == D.CLASS_OIL
    assert D.classify(0.1) == D.CLASS_LOOK_ALIKE
    assert D.classify(0.0) != D.CLASS_OIL_FREE
    assert D.scene_class(0) == D.CLASS_OIL_FREE
    assert D.scene_class(3) is None


# ------------------------------------------------------------------ SC-02 ----


def test_sc02_returns_look_alike_with_a_probability_and_a_breakdown() -> None:
    """§12's P0-CRITICAL scenario: the system refuses to accuse.

    SC-02 is a low-wind dark formation — broad, rounded, at 2.1 m/s. It must come
    back as a look-alike with a real number and the reasoning behind it, not as
    a shrug and not as an accusation.
    """
    result = D.score(_sc02_features())

    assert result.p_class == D.CLASS_LOOK_ALIKE
    assert result.p_oil is not None
    assert result.p_oil < settings.discriminator_oil_threshold
    assert result.factors, "a verdict with no breakdown is not explainable"
    assert result.method == "rule_based"
    assert result.model_version is None

    # The wind gate is the dominant term, and it argues look-alike.
    dominant = result.factors[0]
    assert dominant.feature == "wind_speed_ms"
    assert dominant.direction == "look-alike"

    # The two pixel-derived terms could not be measured and say so.
    for name in ("edge_gradient_mean", "contrast_db"):
        assert name in result.unavailable


def test_sc02_wind_gate_is_violated_and_legible() -> None:
    features = _sc02_features()
    violated, reason = D.wind_gate(features)
    assert violated is True
    assert features.get("wind_speed_ms") < settings.wind_gate_min_ms
    assert "mimics oil" in reason


# ---------------------------------------------------------------- explain ----


def test_contributions_are_additive_against_the_reported_probability() -> None:
    """The panel's bars must sum to the number above them, or it is decoration."""
    result = D.score(
        _features(wind_speed_ms=7.0, shape_complexity=3.2, eccentricity=0.88, area_km2=12.0)
    )
    total = explain.logit(result.base_p_oil) + sum(f.contribution for f in result.factors)
    assert explain.logit(result.p_oil) == pytest.approx(total, abs=1e-9)


def test_the_payload_can_never_omit_the_qualifier() -> None:
    """A rule-based score and a model score look identical as numbers. The label
    is the only thing separating them, so it is not allowed to be optional."""
    payload = explain.to_ui_payload(_sc02_features(), D.score(_sc02_features()))

    assert payload["method"] == "rule_based"
    assert payload["basis"] == "rule_based"
    assert payload["note"], "a probability without its qualifier is unpublishable"
    assert payload["model_version"] is None
    # Top level, beside p_oil — not nested where a panel could miss it.
    assert {"p_oil", "method", "note"} <= set(payload)


def test_the_payload_states_the_wind_gate_with_its_range_and_reason() -> None:
    payload = explain.to_ui_payload(_sc02_features(), D.score(_sc02_features()))
    gate = payload["wind_gate"]

    assert gate["violated"] is True
    assert gate["range_ms"] == [settings.wind_gate_min_ms, settings.wind_gate_max_ms]
    assert "mimics oil" in gate["reason"]


def test_top_k_truncates_by_magnitude_not_by_order() -> None:
    result = D.score(
        _features(
            wind_speed_ms=7.0,
            shape_complexity=4.0,
            eccentricity=0.95,
            area_km2=5.0,
            edge_gradient_mean=1.2,
            contrast_db=-6.0,
        )
    )
    payload = explain.to_ui_payload(_features(wind_speed_ms=7.0), result, top_k=2)
    contributions = [abs(f["contribution"]) for f in payload["factors"]]

    assert len(contributions) == 2
    assert contributions == sorted(contributions, reverse=True)
    assert contributions[0] == max(abs(f.contribution) for f in result.factors)


def test_every_persisted_row_records_its_basis() -> None:
    """`detection_factors.shap` keeps its name, so the row has to say what it is.

    A rule-based contribution sitting in a column called `shap` with nothing
    recording that fact is a landmine for whoever reads the table next.
    """
    result = D.score(_sc02_features())
    rows = explain.factor_rows(result)

    assert rows
    for row in rows:
        assert set(row) == {"feature", "value", "shap", "basis"}
        assert row["basis"] == "rule_based"


def test_method_and_basis_pair_one_to_one() -> None:
    """Two names for the same distinction can only be safe if they are checked."""
    assert D.BASIS_FOR_METHOD == {"lightgbm": "shap", "rule_based": "rule_based"}

    result = D.score(_sc02_features())
    assert result.basis == D.BASIS_FOR_METHOD[result.method]
    for factor in result.factors:
        assert factor.basis == D.BASIS_FOR_METHOD[result.method]


def test_an_unscorable_detection_still_produces_a_payload() -> None:
    """Nothing raises to the user (§8): no score is rendered, not thrown."""
    payload = explain.to_ui_payload(_features(), D.score(_features()))

    assert payload["p_oil"] is None
    assert payload["factors"] == []
    assert "insufficient features" in payload["unavailable"]["p_oil"]
    assert payload["method"] == "rule_based"
    assert payload["wind_gate"]["violated"] is None
    assert payload["wind_gate"]["unavailable"]
