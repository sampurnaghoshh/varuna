"""Whether a detection is allowed to name a vessel at all — the pre-attribution gate.

§5.4 forbids promoting a sub-threshold candidate to an accusation. This is the
step before that one: some detections must not reach the ranking machinery in the
first place, because the thing being traced back is not established to be oil.

The failure it exists to stop is a real one this build produced. SC-01 classified
`look-alike` at P(oil) 0.481 and then went on to attribute a culprit — the system
saying "this is probably not oil" and "that ship spilled it" in the same breath.
Either claim might be defensible alone. Together they are incoherent, and a judge
only has to read two lines of the payload to see it.

SC-02 already refuses on the wind gate. That refusal was written by hand into the
fixture generator, which meant the honesty was in the prose rather than in the
engine — and it covered exactly one scenario. This module is the same principle
made structural: one place, consulted by every path that could attribute, that
can only ever withhold.

WHAT THIS IS NOT
----------------
It is not a second decision band. It never lowers a threshold, never promotes,
and never turns an UNATTRIBUTED frame into an attributed one — `blocks()` returns
a refusal or it returns nothing. A frame that clears this gate still has to clear
ln(10) in `fusion.py` to name anybody, and those thresholds remain untouchable
(§9).

It is also not a quality filter on the drift run. It asks one question about the
detection: does the evidence support calling this oil?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.detection.discriminator import CLASS_OIL, DiscriminatorResult

# Reason codes, so a caller can branch on the cause without matching on prose and
# the UI can render the sentence verbatim (§2.3).
BLOCKED_UNSCORABLE = "unscorable"
BLOCKED_WIND_GATE = "wind_gate"
BLOCKED_NOT_OIL = "not_oil"

RECOMMENDATION_WIND_GATE = (
    "Queue for cross-check against a second pass in a wind window of "
    "{floor:g}–{ceiling:g} m/s."
)
RECOMMENDATION_NOT_OIL = (
    "Queue for cross-check. A dark formation that does not classify as oil is not "
    "traced back to a vessel; re-examine with a second pass or an independent "
    "source before treating it as a discharge."
)
RECOMMENDATION_UNSCORABLE = (
    "Queue for cross-check. The formation could not be scored at all, so there is "
    "nothing to attribute from."
)


@dataclass(frozen=True)
class GateDecision:
    """Whether attribution may run, and — when it may not — why, in words.

    `passed` is the only field a caller may branch on to proceed. `reason` is
    written for the explainability panel and goes on screen unchanged.
    """

    passed: bool
    code: str | None = None
    reason: str = ""
    recommendation: str = ""

    def as_refusal(self, n_vessels_in_frame: int) -> dict[str, Any]:
        """The §7 not-issued attribution payload for a blocked detection.

        Shaped exactly like the one the frontend already renders for SC-02, so a
        withheld verdict has one representation rather than two.
        """
        if self.passed:
            raise ValueError("as_refusal() called on a decision that passed the gate")
        return {
            "issued": False,
            "verdict": "UNATTRIBUTED",
            "blocked_by": self.code,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "candidates": [],
            "n_vessels_in_frame": n_vessels_in_frame,
        }


PASSED = GateDecision(passed=True)


def evaluate(
    result: DiscriminatorResult,
    wind_gate_violated: bool | None,
    wind_gate_reason: str = "",
) -> GateDecision:
    """May this detection be attributed to a vessel?

    Checked in order of how fundamental the objection is, so the reason a caller
    shows is the most basic one that applies:

      1. The formation could not be scored at all — there is no classification to
         act on, and a probability from an empty term set is not one.
      2. The wind gate is violated — the §5.2 validity condition. Below 3 m/s the
         sea surface itself mimics oil and above 12 m/s slicks disperse below
         detectability, so the detection is not trustworthy evidence whatever its
         probability says.
      3. It did not classify as oil.

    A `None` wind gate means no wind was available to evaluate it. That is not
    treated as a violation — an unmeasured gate is not a failed one — and rule 3
    still has to pass on its own.
    """
    if result.p_oil is None or result.p_class is None:
        return GateDecision(
            passed=False,
            code=BLOCKED_UNSCORABLE,
            reason=(
                "No attribution issued: the dark formation could not be scored, so "
                "there is no basis for calling it oil. "
                + result.unavailable.get("p_oil", "")
            ).strip(),
            recommendation=RECOMMENDATION_UNSCORABLE,
        )

    if wind_gate_violated:
        return GateDecision(
            passed=False,
            code=BLOCKED_WIND_GATE,
            reason=(
                f"No attribution issued: {wind_gate_reason} It classifies "
                f"{result.p_class} at P(oil) {result.p_oil:.3f} — {result.method} — "
                "and the gate alone would withhold attribution regardless of what "
                "that number said (§5.2)."
            ),
            recommendation=RECOMMENDATION_WIND_GATE.format(
                floor=settings.wind_gate_min_ms, ceiling=settings.wind_gate_max_ms
            ),
        )

    if result.p_class != CLASS_OIL:
        return GateDecision(
            passed=False,
            code=BLOCKED_NOT_OIL,
            reason=(
                f"No attribution issued: the formation classifies {result.p_class} at "
                f"P(oil) {result.p_oil:.3f}, below the {settings.discriminator_oil_threshold:g} "
                f"oil threshold ({result.method}). A detection that does not read as oil "
                "is not traced back to a vessel — naming a culprit for it would assert "
                "a discharge the detector itself does not support."
            ),
            recommendation=RECOMMENDATION_NOT_OIL,
        )

    return PASSED
