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

Two structural consequences worth stating, because both are load-bearing:

  * `FrameResult.culprit` is the only way to ask this module who did it, and it
    returns None unless the top candidate cleared MODERATE. A sub-threshold
    vessel has no route to becoming an accusation.
  * A vessel with no evidence at all - no origin overlap, so no t* for E2/E3/E4
    to anchor to - is not ranked. It goes to `unrankable` with no log LR, no rank
    and no posterior, because the only thing left to rank it on would be its
    prior, and a prior is not evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from app.attribution.channels import ChannelScores
from app.config import settings

LOG_LR_MODERATE: Final[float] = math.log(10.0)
LOG_LR_STRONG: Final[float] = math.log(100.0)
NONE_OF_THE_ABOVE_LOG_LR: Final[float] = math.log(10.0)

# A channel score of exactly zero is a real inference - no measurable overlap, or
# a vessel not underway - but -inf is not a number a ranking can carry. The floor
# is stated rather than hidden: a term at the floor means "no measurable signal",
# and every candidate it applies to lands far below ln(10) regardless of its
# exact value, so it can move no verdict.
SCORE_FLOOR: Final[float] = 1e-6

INSUFFICIENT_EVIDENCE: Final[str] = "insufficient evidence"

Verdict = Literal["STRONG", "MODERATE", "UNATTRIBUTED"]


@dataclass(frozen=True)
class Candidate:
    mmsi: int
    log_lr: float
    lr: float
    posterior: float
    verdict: Verdict
    rank: int
    prior: float
    channels: ChannelScores
    # Per-channel contribution in nats, for the §7 channel breakdown. None where
    # the channel had no score and therefore contributed nothing.
    terms: dict[str, float | None]


@dataclass(frozen=True)
class Unrankable:
    """A vessel in frame that evidence cannot speak to.

    Carries no log LR, no rank and no posterior - not as an omission but as the
    point: there is no field here for a caller to mistake for a score.
    """

    mmsi: int
    reason: str
    channels: ChannelScores
    verdict: Verdict = "UNATTRIBUTED"


@dataclass(frozen=True)
class FrameResult:
    candidates: list[Candidate]
    unrankable: list[Unrankable]
    none_posterior: float
    background: tuple[float, float, float]

    @property
    def culprit(self) -> Candidate | None:
        """The accused vessel, or None when the evidence does not reach a verdict.

        The single place downstream code asks "who did it". It cannot answer with
        a sub-threshold candidate, which is what §5.4 means by UNATTRIBUTED never
        returning the top-ranked vessel as a culprit.
        """
        if not self.candidates:
            return None
        top = self.candidates[0]
        return top if top.verdict in ("STRONG", "MODERATE") else None

    @property
    def verdict(self) -> Verdict:
        """The frame's verdict — UNATTRIBUTED unless a culprit cleared a band."""
        culprit = self.culprit
        return culprit.verdict if culprit is not None else "UNATTRIBUTED"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def background_scores(
    all_scores: Sequence[ChannelScores],
) -> tuple[float, float, float]:
    """Mean s1, s2, s3 across every vessel in the frame.

    s1 is a share of the frame's total corridor mass, so it is averaged over
    every vessel - including those that scored zero, which is a real observation
    about that vessel and belongs in the denominator. s2 and s3 average only over
    vessels where the channel had a score: a missing COG says nothing about the
    frame's typical axial coherence and must not drag the background down.
    """
    s1 = _mean([s.s1 for s in all_scores])
    s2 = _mean([s.s2 for s in all_scores if s.s2 is not None])
    s3 = _mean([s.s3 for s in all_scores if s.s3 is not None])
    return s1, s2, s3


def _log_ratio(score: float | None, background: float, weight: float) -> float | None:
    """One weighted log ratio in nats, or None when the channel had no score."""
    if score is None:
        return None
    if background <= 0.0:
        return 0.0
    return weight * math.log(max(score, SCORE_FLOOR) / max(background, SCORE_FLOOR))


def log_likelihood_ratio(
    scores: ChannelScores,
    background: tuple[float, float, float],
    prior: float,
    weights: tuple[float, float, float] | None = None,
) -> tuple[float, dict[str, float | None]]:
    """log LR_j and its per-channel terms, in nats (§5.4).

    An unavailable channel contributes zero nats. That is the neutral element of
    a log-ratio sum, so absent evidence neither accuses nor exonerates - it just
    is not counted.
    """
    weights = settings.fusion_weights if weights is None else weights
    s1_bg, s2_bg, s3_bg = background

    terms: dict[str, float | None] = {
        "s1": _log_ratio(scores.s1, s1_bg, weights[0]),
        "s2": _log_ratio(scores.s2, s2_bg, weights[1]),
        "s3": _log_ratio(scores.s3, s3_bg, weights[2]),
        # E4 and the prior are not ratios against the frame: E4 is a stated boost
        # in [1.0, 2.2] and the prior is already mean-normalised (see priors.py).
        "s4": None if scores.s4 is None else math.log(max(scores.s4, SCORE_FLOOR)),
        "prior": math.log(max(prior, SCORE_FLOOR)),
    }
    return sum(t for t in terms.values() if t is not None), terms


def verdict_for(log_lr: float) -> Verdict:
    """>= ln(100) STRONG, >= ln(10) MODERATE, otherwise UNATTRIBUTED."""
    if log_lr >= LOG_LR_STRONG:
        return "STRONG"
    if log_lr >= LOG_LR_MODERATE:
        return "MODERATE"
    return "UNATTRIBUTED"


def posteriors(log_lrs: Mapping[int, float]) -> dict[int, float]:
    """Softmax including the pinned none-of-the-above hypothesis."""
    posterior, _ = _softmax_with_none(log_lrs)
    return posterior


def none_posterior(log_lrs: Mapping[int, float]) -> float:
    """P(none of these vessels), from the same softmax.

    Exposed separately because the UI has to be able to render §2.4 honestly: a
    frame where "none of the above" holds most of the posterior is a frame the
    system is saying it cannot attribute, and that has to be visible.
    """
    _, none = _softmax_with_none(log_lrs)
    return none


def _softmax_with_none(log_lrs: Mapping[int, float]) -> tuple[dict[int, float], float]:
    logits = list(log_lrs.values()) + [NONE_OF_THE_ABOVE_LOG_LR]
    peak = max(logits)
    weights = {mmsi: math.exp(value - peak) for mmsi, value in log_lrs.items()}
    none_weight = math.exp(NONE_OF_THE_ABOVE_LOG_LR - peak)
    total = sum(weights.values()) + none_weight
    return {mmsi: w / total for mmsi, w in weights.items()}, none_weight / total


def rank_candidates(
    log_lrs: Mapping[int, float],
    terms: Mapping[int, dict[str, float | None]],
    priors: Mapping[int, float],
    scores: Mapping[int, ChannelScores],
) -> list[Candidate]:
    """Ranked descending by log LR, each with its own verdict.

    Rank is an ordering, not an accusation: rank 1 with an UNATTRIBUTED verdict
    is a normal, correct output. Only `FrameResult.culprit` turns a ranking into
    a claim, and only above ln(10).
    """
    posterior = posteriors(log_lrs)
    ordered = sorted(log_lrs.items(), key=lambda item: (-item[1], item[0]))
    return [
        Candidate(
            mmsi=mmsi,
            log_lr=log_lr,
            lr=math.exp(log_lr),
            posterior=posterior[mmsi],
            verdict=verdict_for(log_lr),
            rank=index + 1,
            prior=priors[mmsi],
            channels=scores[mmsi],
            terms=terms[mmsi],
        )
        for index, (mmsi, log_lr) in enumerate(ordered)
    ]


def fuse(
    scores: Mapping[int, ChannelScores],
    priors: Mapping[int, float],
    weights: tuple[float, float, float] | None = None,
) -> FrameResult:
    """Combine channels and priors into a ranked frame result (§5.4).

    `priors` must already be mean-normalised for the frame - see
    `priors.frame_priors`. Vessels with no evidence are separated out before the
    background means are taken: they contribute no s2/s3 anyway, and including a
    vessel that was never observed in the s1 denominator would dilute the
    background with a non-observation.
    """
    rankable = {mmsi: s for mmsi, s in scores.items() if s.has_evidence}
    unrankable = [
        Unrankable(mmsi=mmsi, reason=INSUFFICIENT_EVIDENCE, channels=s)
        for mmsi, s in sorted(scores.items())
        if not s.has_evidence
    ]

    background = background_scores(list(rankable.values()))

    log_lrs: dict[int, float] = {}
    terms: dict[int, dict[str, float | None]] = {}
    for mmsi, channel_scores in rankable.items():
        log_lr, channel_terms = log_likelihood_ratio(
            channel_scores, background, priors.get(mmsi, 1.0), weights
        )
        log_lrs[mmsi] = log_lr
        terms[mmsi] = channel_terms

    candidates = rank_candidates(
        log_lrs, terms, {mmsi: priors.get(mmsi, 1.0) for mmsi in log_lrs}, rankable
    )
    return FrameResult(
        candidates=candidates,
        unrankable=unrankable,
        none_posterior=none_posterior(log_lrs) if log_lrs else 1.0,
        background=background,
    )
