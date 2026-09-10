"""Vessel-type and history priors — §5.4.

Normalised: tanker 3.0, bulk/cargo 1.5, fishing 1.0, other 1.0, passenger 0.5,
each multiplied by (1 + 0.5 * prior_detections_j). Values live in config.py.

The alias table below is not a tunable and deliberately stays out of config.py:
it is the AIS vocabulary, not a number anyone should be turning. The weights it
maps onto are the tunables, and those are in `settings.vessel_type_priors`.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

OTHER = "other"

# Danish AIS ships free text ("Crude Oil Tanker", "Cargo", "Sailing", "Undefined"),
# so match on substrings rather than on equality. Order matters: the first hit
# wins, and "tanker" must be tested before "cargo" for a "Cargo/Tanker" entry.
_ALIASES: tuple[tuple[str, str], ...] = (
    ("tanker", "tanker"),
    ("cargo", "cargo"),
    ("bulk", "bulk"),
    ("container", "cargo"),
    ("fishing", "fishing"),
    ("trawler", "fishing"),
    ("passenger", "passenger"),
    ("ferry", "passenger"),
    ("cruise", "passenger"),
)


def canonical_type(vessel_type: str | None) -> str:
    """AIS free text to one of the §5.4 prior classes."""
    if not vessel_type:
        return OTHER
    text = vessel_type.strip().lower()
    for needle, canonical in _ALIASES:
        if needle in text:
            return canonical
    return OTHER


def type_prior(vessel_type: str | None) -> float:
    """Tanker 3.0, bulk/cargo 1.5, fishing 1.0, other 1.0, passenger 0.5 (§5.4)."""
    return settings.vessel_type_priors.get(canonical_type(vessel_type), settings.prior_other)


def history_multiplier(prior_detections: int) -> float:
    """1 + 0.5 * prior_detections — a repeat offender is likelier, not guilty."""
    if prior_detections < 0:
        raise ValueError(f"prior_detections must be non-negative, got {prior_detections}")
    return 1.0 + settings.prior_history_coefficient * prior_detections


def prior(vessel_type: str | None, prior_detections: int) -> float:
    return type_prior(vessel_type) * history_multiplier(prior_detections)


def normalise(priors: dict[int, float]) -> dict[int, float]:
    """Keyed by MMSI. Normalised by the frame MEAN, not by the sum.

    Every other term in §5.4 is a ratio against what an average vessel in this
    frame scores, so a vessel with no distinguishing feature contributes zero
    nats. The prior has to be built the same way or it is not commensurate with
    the rest of the sum: dividing by the sum instead would subtract roughly
    log(N) from every candidate, so a slick in a busy shipping lane would score
    lower than the identical slick in an empty sea and the ln(10) / ln(100) bands
    would move with traffic density. Those bands are a safety property (§9) and
    nothing may move them, arithmetic included.

    A mean-normalised prior means: this vessel is 1.7x as prior-likely as the
    average vessel in the frame - which is what the log LR is entitled to add.
    """
    if not priors:
        return {}
    mean = sum(priors.values()) / len(priors)
    if mean <= 0.0:
        raise ValueError("Frame priors must be positive; got a non-positive mean")
    return {mmsi: value / mean for mmsi, value in priors.items()}


def frame_priors(vessels: dict[int, tuple[str | None, int]]) -> dict[int, float]:
    """Mean-normalised priors for a whole frame, from {mmsi: (type, detections)}."""
    return normalise({mmsi: prior(t, n) for mmsi, (t, n) in vessels.items()})
