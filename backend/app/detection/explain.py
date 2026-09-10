"""SHAP contributions to the UI explainability payload — §10 P0-3.

Rows persist to `detection_factors` (§6) and drive the explainability panel. The
wind-gate flag surfaces alongside them: it is one of the strongest answers we
have in Q&A (§5.2).
"""

from typing import Any

from app.detection.features import DetectionFeatures


def shap_values(model: object, features: DetectionFeatures) -> dict[str, float]: ...


def to_ui_payload(
    features: DetectionFeatures,
    shap: dict[str, float],
    wind_gate_violated: bool,
    top_k: int,
) -> dict[str, Any]: ...
