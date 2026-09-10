"""LightGBM oil vs look-alike classifier — §10 P0-3.

Trained on Part III indices 001-120, validated on 121-145, never on the holdout
range 146-150 (§0, §9). Reported metrics carry their split and n (§16).
"""

from pathlib import Path

from app.detection.features import DetectionFeatures


def load_model(model_path: Path): ...


def predict_p_oil(model: object, features: DetectionFeatures) -> float: ...


def classify(p_oil: float, threshold: float) -> str:
    """Returns one of 'oil', 'look-alike', 'oil-free' — the `detections.class`
    domain (§6)."""
    ...
