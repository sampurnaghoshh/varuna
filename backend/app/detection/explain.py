"""SHAP contributions to the UI explainability payload — §10 P0-3.

Rows persist to `detection_factors` (§6) and drive the explainability panel. The
wind-gate flag surfaces alongside them: it is one of the strongest answers we
have in Q&A (§5.2).

Two labels travel with every payload and every persisted row, and neither is
optional:

    method   how P(oil) was produced      lightgbm | rule_based
    basis    where a contribution came from   shap  | rule_based

They pair one-to-one. `detection_factors` keeps its column named `shap` — a
rename would be a schema migration for a cosmetic gain — so `basis` sits on the
row beside it. A rule-based contribution stored in a column called `shap` with
nothing recording that fact is a landmine for whoever reads the table next.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.config import settings
from app.detection.discriminator import (
    BASIS_FOR_METHOD,
    DiscriminatorResult,
    Factor,
    LoadedModel,
)
from app.detection.features import ExtractedFeatures, wind_gate_reason, wind_gate_violated

logger = logging.getLogger(__name__)

SHAP_UNAVAILABLE = "SHAP could not explain this prediction: {error}"

# Physics rationale is what the panel shows next to a bare SHAP number. The
# rule-based path carries its own; these cover the model path.
SHAP_RATIONALE = "SHAP contribution of this feature to the model's P(oil)"


def shap_values(model: object, features: ExtractedFeatures) -> dict[str, float]:
    """Per-feature SHAP contributions in log-odds, for a loaded booster.

    Returns an empty mapping rather than raising when SHAP is unavailable: an
    unexplained prediction still has a P(oil), and the panel says the
    explanation is missing rather than the stage failing (§8).
    """
    if not isinstance(model, LoadedModel):
        raise TypeError("shap_values expects the LoadedModel returned by load_model")

    try:
        import shap

        row = [
            float(value) if (value := features.get(name)) is not None else float("nan")
            for name in model.feature_names
        ]
        explainer = shap.TreeExplainer(model.booster)
        contributions = explainer.shap_values([row])[0]
        return {
            name: float(contribution)
            for name, contribution in zip(model.feature_names, contributions, strict=True)
        }
    except (ImportError, ValueError, TypeError) as error:
        logger.warning("SHAP unavailable for %s: %s", model.model_version, error)
        return {}


def factors_from_shap(
    features: ExtractedFeatures, shap: dict[str, float], top_k: int | None = None
) -> list[Factor]:
    """SHAP contributions as `Factor` rows — the same shape the rules produce."""
    top_k = settings.discriminator_top_k_factors if top_k is None else top_k
    factors = [
        Factor(
            feature=name,
            value=features.get(name),
            contribution=contribution,
            rationale=SHAP_RATIONALE,
            basis="shap",
        )
        for name, contribution in shap.items()
    ]
    factors.sort(key=lambda factor: abs(factor.contribution), reverse=True)
    return factors[:top_k]


def factor_rows(result: DiscriminatorResult) -> list[dict[str, Any]]:
    """`detection_factors` rows — the §6 tuple, with `basis` on every one.

    Built here rather than at the insert site so a writer cannot forget the
    column: the row shape and the label that qualifies it are produced together.
    """
    return [
        {
            "feature": factor.feature,
            "value": factor.value,
            "shap": factor.contribution,
            "basis": factor.basis,
        }
        for factor in result.factors
    ]


def to_ui_payload(
    features: ExtractedFeatures,
    result: DiscriminatorResult,
    top_k: int | None = None,
) -> dict[str, Any]:
    """The explainability panel's payload for one detection.

    `method` and `note` sit at the top level beside `p_oil`, not nested where a
    panel could render the probability without them. That placement is the point:
    a rule-based score and a trained-model score look identical as numbers, and
    only the label distinguishes them (§2.2).
    """
    top_k = settings.discriminator_top_k_factors if top_k is None else top_k
    wind = features.get("wind_speed_ms")
    gate_violated = wind_gate_violated(wind) if wind is not None else None

    factors = sorted(result.factors, key=lambda factor: abs(factor.contribution), reverse=True)
    payload: dict[str, Any] = {
        "p_oil": result.p_oil,
        "class": result.p_class,
        "method": result.method,
        "basis": BASIS_FOR_METHOD[result.method],
        "note": result.note,
        "model_version": result.model_version,
        "base_p_oil": result.base_p_oil,
        "evidence_fraction": result.evidence_fraction,
        "factors": [
            {
                "feature": factor.feature,
                "value": factor.value,
                "contribution": factor.contribution,
                "direction": factor.direction,
                "rationale": factor.rationale,
                "basis": factor.basis,
            }
            for factor in factors[:top_k]
        ],
        "wind_gate": {
            "violated": gate_violated,
            "wind_speed_ms": wind,
            "range_ms": [settings.wind_gate_min_ms, settings.wind_gate_max_ms],
            "reason": wind_gate_reason(wind),
        },
        "unavailable": dict(result.unavailable),
    }
    if gate_violated is None:
        payload["wind_gate"]["unavailable"] = wind_gate_reason(None)
    return payload


def logit(p: float) -> float:
    """Inverse of the logistic — the additive scale the factors live on."""
    return math.log(p / (1.0 - p))
