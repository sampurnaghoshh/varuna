"""LightGBM oil vs look-alike classifier — §10 P0-3.

Trained on Part III indices 001-120, validated on 121-145, never on the holdout
range 146-150 (§0, §9). Reported metrics carry their split and n (§16).

No model is trained in this build — Zenodo Part III has not landed, so there is
no training data. `score()` therefore falls back to a deterministic rule-based
scorer built from the §5.2 physics, and says so: every result carries
`method` and a `note`, and a rule-based result never claims a `model_version`.
A rule-based scorer is defensible; a rule-based scorer presented as a trained
model is the thing §2.2 exists to prevent.

The rule weights in `config.py` are set from physics rationale alone. None is
fitted, and none was adjusted to reproduce a figure from the spec — when the
number this produces disagreed with §12's illustrative "P(oil) ~ 0.12", §12 was
amended to match the engine rather than the weights tuned to match §12.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.detection.features import (
    ExtractedFeatures,
    wind_gate_reason,
    wind_gate_violated,
)

logger = logging.getLogger(__name__)

Method = Literal["lightgbm", "rule_based"]
Basis = Literal["shap", "rule_based"]

CLASS_OIL = "oil"
CLASS_LOOK_ALIKE = "look-alike"
CLASS_OIL_FREE = "oil-free"

# `method` and `basis` answer different questions and pair one-to-one. This map
# is the pairing, so the two can never drift into disagreeing about one run.
BASIS_FOR_METHOD: dict[Method, Basis] = {"lightgbm": "shap", "rule_based": "rule_based"}

RULE_BASED_NOTE = (
    "No trained discriminator is present in this build, so P(oil) comes from a "
    "deterministic rule-based scorer over the §5.2 physics — the wind gate, "
    "shape complexity, eccentricity and slick area — not from a trained model. "
    "It carries no validation metric, no split and no n, because there is no "
    "model to have validated."
)
LIGHTGBM_NOTE = (
    "P(oil) is a trained LightGBM discriminator output; factor contributions are "
    "SHAP values. Metrics quoted for this model come from the validation split "
    "and carry their n (§16)."
)
INSUFFICIENT_FEATURES = (
    "insufficient features: none of the wind gate, shape complexity, "
    "eccentricity or area was available, so there is no evidence to score. A "
    "probability from an empty term set is the base rate wearing a "
    "measurement's clothes (§2.2)."
)
NO_MODEL_VERSION = "no trained model ran; a rule-based score has no model version"

# The scorer refuses to produce a number unless at least one of these resolved.
# The two image-derived terms are deliberately not on the list: they are absent
# for the whole of this build, and requiring them would mean no score at all.
REQUIRED_TERMS: tuple[str, ...] = (
    "wind_speed_ms",
    "shape_complexity",
    "eccentricity",
    "area_km2",
)


@dataclass(frozen=True)
class Factor:
    """One additive log-odds contribution, with the physics behind it.

    The rule-based path and the SHAP path produce the same shape, so the
    explainability panel renders either without knowing which it has — while
    `basis` still records which it was.
    """

    feature: str
    value: float | None
    contribution: float
    rationale: str
    basis: Basis

    @property
    def direction(self) -> Literal["oil", "look-alike", "neutral"]:
        if self.contribution > 0.0:
            return "oil"
        if self.contribution < 0.0:
            return "look-alike"
        return "neutral"


@dataclass(frozen=True)
class DiscriminatorResult:
    """P(oil) for one dark formation, and how it was arrived at.

    `p_oil` is None when the evidence was too thin to score at all — that is a
    stated outcome, not an error, and the caller reports it as such.
    """

    p_oil: float | None
    p_class: str | None
    method: Method
    note: str
    model_version: str | None
    factors: list[Factor] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)
    base_p_oil: float = 0.0
    evidence_fraction: float = 0.0

    @property
    def basis(self) -> Basis:
        return BASIS_FOR_METHOD[self.method]


# --------------------------------------------------------------- model path ----


@dataclass(frozen=True)
class LoadedModel:
    """A LightGBM booster plus the §16 provenance it must be quoted with."""

    booster: Any
    feature_names: tuple[str, ...]
    model_version: str
    split: str | None = None
    n_val: int | None = None


def load_model(model_path: Path) -> LoadedModel | None:
    """Load a booster and its sidecar, or return None. Never raises (§8).

    A missing model is the normal case in this build, not a failure: the caller
    falls back to the rule-based scorer and labels the result accordingly.
    """
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(".json")
    if not model_path.exists():
        logger.info("no discriminator at %s; rule-based scoring will be used", model_path)
        return None

    try:
        import lightgbm

        booster = lightgbm.Booster(model_file=str(model_path))
        metadata: dict[str, Any] = (
            json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        )
        return LoadedModel(
            booster=booster,
            feature_names=tuple(metadata.get("feature_names") or booster.feature_name()),
            model_version=str(metadata.get("model_version", model_path.stem)),
            split=metadata.get("split"),
            n_val=metadata.get("n_val"),
        )
    except (OSError, ValueError, KeyError, ImportError) as error:
        logger.warning("discriminator at %s could not be loaded: %s", model_path, error)
        return None


def predict_p_oil(model: object, features: ExtractedFeatures) -> float:
    """P(oil) from a loaded booster.

    A feature the model was trained on but that is unavailable here is passed as
    NaN rather than as an imputed value: LightGBM splits on missingness
    natively, and an imputed mean would be a number the scene did not produce.
    """
    if not isinstance(model, LoadedModel):
        raise TypeError("predict_p_oil expects the LoadedModel returned by load_model")

    row = [
        float(value) if (value := features.get(name)) is not None else float("nan")
        for name in model.feature_names
    ]
    prediction = model.booster.predict([row])
    return float(prediction[0])


def classify(p_oil: float, threshold: float | None = None) -> str:
    """Returns one of 'oil', 'look-alike', 'oil-free' — the `detections.class`
    domain (§6).

    A detection that exists is one of the first two: something dark was found, so
    the question is only whether it is oil. 'oil-free' is the scene-level value
    for a scene that yielded no detections at all — SC-03 (§12) — and is reached
    through `scene_class`, not through a probability.
    """
    threshold = settings.discriminator_oil_threshold if threshold is None else threshold
    return CLASS_OIL if p_oil >= threshold else CLASS_LOOK_ALIKE


def scene_class(n_detections: int) -> str | None:
    """'oil-free' for a scene with no dark formations; None once there are some.

    A scene with detections takes its class from those detections, not from here.
    """
    return CLASS_OIL_FREE if n_detections == 0 else None


# ---------------------------------------------------------- rule-based path ----


def _soft(value: float, pivot: float, scale: float) -> float:
    """Signed evidence in [-1, 1] for how far `value` sits past `pivot`.

    tanh rather than a step: a formation just past the pivot is weak evidence
    and one far past it is strong, and neither should flip on a rounding error.
    """
    return math.tanh((value - pivot) / scale)


def _wind_term(wind_speed_ms: float) -> tuple[float, str]:
    """The §5.2 gate, as a step rather than a ramp.

    §5.2 defines the gate as a binary validity flag, so this reproduces that
    rather than inventing a continuum the spec does not have. The low-wind
    penalty is the larger of the two: below the floor the sea surface itself
    mimics oil, which argues the formation is not a slick at all, while above
    the ceiling the formation is real and merely unlikely to still be oil.
    """
    if wind_speed_ms < settings.wind_gate_min_ms:
        return -settings.rule_w_wind_gate_low, wind_gate_reason(wind_speed_ms)
    if wind_speed_ms > settings.wind_gate_max_ms:
        return -settings.rule_w_wind_gate_high, wind_gate_reason(wind_speed_ms)
    return settings.rule_w_wind_in_band, wind_gate_reason(wind_speed_ms)


def _rule_terms(features: ExtractedFeatures) -> tuple[list[Factor], float, float]:
    """Every rule term that could be evaluated, plus the weight seen and missed.

    A term whose feature is unavailable is skipped, not zeroed: a zero
    contribution is the statement "this feature argues neither way", which is a
    measurement we did not make (§2.2). The weight it would have carried is
    returned so the caller can shrink the result toward the base rate in
    proportion to the evidence actually available.
    """
    factors: list[Factor] = []
    weight_seen = 0.0
    weight_total = 0.0

    def add(name: str, value: Any, weight: float, contribution: float, rationale: str) -> None:
        nonlocal weight_seen, weight_total
        weight_total += weight
        if value is None:
            return
        weight_seen += weight
        factors.append(
            Factor(
                feature=name,
                value=float(value),
                contribution=contribution,
                rationale=rationale,
                basis="rule_based",
            )
        )

    wind = features.get("wind_speed_ms")
    wind_contribution, wind_rationale = _wind_term(wind) if wind is not None else (0.0, "")
    add("wind_speed_ms", wind, settings.rule_w_wind_gate_low, wind_contribution, wind_rationale)

    complexity = features.get("shape_complexity")
    add(
        "shape_complexity",
        complexity,
        settings.rule_w_shape_complexity,
        settings.rule_w_shape_complexity
        * _soft(
            complexity or 0.0,
            settings.rule_shape_complexity_pivot,
            settings.rule_shape_complexity_scale,
        ),
        "real slicks are sheared by the drift that moved them and read as "
        "filamentary; low-wind cells and biogenic films stay broad and rounded",
    )

    eccentricity = features.get("eccentricity")
    add(
        "eccentricity",
        eccentricity,
        settings.rule_w_eccentricity,
        settings.rule_w_eccentricity
        * _soft(
            eccentricity or 0.0,
            settings.rule_eccentricity_pivot,
            settings.rule_eccentricity_scale,
        ),
        "a deliberate discharge while underway lays oil along the track, so it is "
        "elongated; a weather feature has no such axis",
    )

    edge = features.get("edge_gradient_mean")
    add(
        "edge_gradient_mean",
        edge,
        settings.rule_w_edge_gradient,
        settings.rule_w_edge_gradient
        * _soft(
            edge or 0.0,
            settings.rule_edge_gradient_pivot_db,
            settings.rule_edge_gradient_scale_db,
        ),
        "oil damps capillary waves sharply at its boundary; a low-wind cell fades "
        "into the surrounding sea",
    )

    contrast = features.get("contrast_db")
    add(
        "contrast_db",
        contrast,
        settings.rule_w_contrast,
        settings.rule_w_contrast
        * _soft(
            -(contrast or 0.0),
            settings.rule_contrast_pivot_db,
            settings.rule_contrast_scale_db,
        ),
        "oil sits 3-10 dB below the surrounding sea; a marginal look-alike "
        "manages 1-3 dB",
    )

    area = features.get("area_km2")
    area_band = 0.0
    if area is not None and area > 0.0:
        offset = abs(math.log10(area) - settings.rule_area_log10_centre)
        area_band = settings.rule_w_area * (
            1.0 - 2.0 * min(offset / settings.rule_area_log10_half_width, 1.0)
        )
    add(
        "area_km2",
        area,
        settings.rule_w_area,
        area_band,
        "scored as a band, not a direction: below it a formation is speckle, "
        "above it the footprint belongs to weather — a wind shadow spans "
        "hundreds of km², a discharge does not",
    )

    return factors, weight_seen, weight_total


def score_rule_based(features: ExtractedFeatures) -> DiscriminatorResult:
    """P(oil) from the §5.2 physics, with no model involved.

    The result is shrunk toward the base rate in proportion to the fraction of
    rule weight actually observed. With two of six terms unmeasurable in this
    build, asserting the full deviation would claim a confidence the evidence
    does not carry — less evidence has to mean less certainty, in both
    directions.
    """
    factors, weight_seen, weight_total = _rule_terms(features)
    unavailable = dict(features.unavailable)

    scored = {factor.feature for factor in factors}
    if not scored & set(REQUIRED_TERMS):
        return DiscriminatorResult(
            p_oil=None,
            p_class=None,
            method="rule_based",
            note=RULE_BASED_NOTE,
            model_version=None,
            factors=[],
            unavailable={
                **unavailable,
                "p_oil": INSUFFICIENT_FEATURES,
                "class": INSUFFICIENT_FEATURES,
                "model_version": NO_MODEL_VERSION,
            },
            base_p_oil=settings.rule_base_p_oil,
            evidence_fraction=0.0,
        )

    evidence_fraction = weight_seen / weight_total if weight_total > 0.0 else 0.0
    # Shrink each contribution rather than the total, so the factors the panel
    # shows still sum to the log-odds the probability came from.
    shrunk = [
        Factor(
            feature=factor.feature,
            value=factor.value,
            contribution=factor.contribution * evidence_fraction,
            rationale=factor.rationale,
            basis=factor.basis,
        )
        for factor in factors
    ]

    base = settings.rule_base_p_oil
    logit = math.log(base / (1.0 - base)) + sum(factor.contribution for factor in shrunk)
    p_oil = 1.0 / (1.0 + math.exp(-logit))

    shrunk.sort(key=lambda factor: abs(factor.contribution), reverse=True)
    unavailable.setdefault("model_version", NO_MODEL_VERSION)

    return DiscriminatorResult(
        p_oil=p_oil,
        p_class=classify(p_oil),
        method="rule_based",
        note=RULE_BASED_NOTE,
        model_version=None,
        factors=shrunk,
        unavailable=unavailable,
        base_p_oil=base,
        evidence_fraction=evidence_fraction,
    )


def score(features: ExtractedFeatures, model: LoadedModel | None = None) -> DiscriminatorResult:
    """P(oil) for one dark formation — the entry point every caller uses.

    With a trained model this is the model's output and the factors are SHAP
    values; without one it is the rule-based scorer. Which of the two ran is
    always stated on the result, never inferred by the caller.
    """
    if model is None:
        return score_rule_based(features)

    p_oil = predict_p_oil(model, features)
    return DiscriminatorResult(
        p_oil=p_oil,
        p_class=classify(p_oil),
        method="lightgbm",
        note=LIGHTGBM_NOTE,
        model_version=model.model_version,
        factors=[],  # populated by explain.shap_values — this module does not import shap
        unavailable=dict(features.unavailable),
        base_p_oil=settings.rule_base_p_oil,
        evidence_fraction=1.0,
    )


def wind_gate(features: ExtractedFeatures) -> tuple[bool | None, str]:
    """The §5.2 gate flag and its reason, for the result the caller assembles."""
    wind = features.get("wind_speed_ms")
    if wind is None:
        return None, wind_gate_reason(None)
    return wind_gate_violated(wind), wind_gate_reason(wind)
