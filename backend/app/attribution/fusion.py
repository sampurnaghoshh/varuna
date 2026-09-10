"""Bayesian likelihood-ratio combination and decision bands — §5.4.

    log LR_j = w1*log(s1_j/s1_bg) + w2*log(s2_j/s2_bg) + w3*log(s3_j/s3_bg)
               + log(s4_j) + log(pi_j),      w = [1.0, 0.7, 0.5]

Background terms s_bg are the mean channel scores across all vessels in the
frame. That is what makes this a likelihood *ratio* rather than an arbitrary
score, and it is why the number survives cross-examination.

Posterior is a softmax over log LR_j including an explicit "none of the above"
hypothesis pinned at ln(10).

The thresholds below are a safety property, not a tuning parameter (§9). They
are module-level constants rather than settings fields precisely so that no
environment variable, request parameter, or demo flag can move them. No code
path may promote a sub-threshold candidate to an accusation (§5.4).
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

LOG_LR_MODERATE: Final[float] = math.log(10.0)
LOG_LR_STRONG: Final[float] = math.log(100.0)
NONE_OF_THE_ABOVE_LOG_LR: Final[float] = math.log(10.0)

Verdict = Literal["STRONG", "MODERATE", "UNATTRIBUTED"]


@dataclass(frozen=True)
class Candidate:
    mmsi: int
    log_lr: float
    lr: float
    posterior: float
    verdict: Verdict
    rank: int


def background_scores(
    all_scores: list[tuple[float, float, float, float]],
) -> tuple[float, float, float]:
    """Mean s1, s2, s3 across every vessel in the frame."""
    ...


def log_likelihood_ratio(
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    background: tuple[float, float, float],
    prior: float,
    weights: tuple[float, float, float],
) -> float: ...


def verdict_for(log_lr: float) -> Verdict:
    """>= ln(100) STRONG, >= ln(10) MODERATE, otherwise UNATTRIBUTED."""
    ...


def posteriors(log_lrs: dict[int, float]) -> dict[int, float]:
    """Softmax including the pinned none-of-the-above hypothesis."""
    ...


def rank_candidates(log_lrs: dict[int, float]) -> list[Candidate]: ...
