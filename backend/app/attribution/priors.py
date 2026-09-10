"""Vessel-type and history priors — §5.4.

Normalised: tanker 3.0, bulk/cargo 1.5, fishing 1.0, other 1.0, passenger 0.5,
each multiplied by (1 + 0.5 * prior_detections_j). Values live in config.py.
"""


def type_prior(vessel_type: str | None) -> float: ...


def history_multiplier(prior_detections: int) -> float: ...


def prior(vessel_type: str | None, prior_detections: int) -> float: ...


def normalise(priors: dict[int, float]) -> dict[int, float]:
    """Keyed by MMSI."""
    ...
