"""The pre-attribution gate — `app/attribution/gate.py`.

SC-01 shipped a payload that classified `look-alike` at P(oil) 0.481 and named a
culprit in the same breath. The gate exists so that cannot be expressed.

The load-bearing property is one-directional: this module may withhold and may
never promote. Everything else here is a consequence of that.
"""

from __future__ import annotations

import pytest

from app.attribution import gate
from app.config import settings
from app.detection import discriminator as D
from app.detection import features as F


def _result(
    p_oil: float | None,
    p_class: str | None,
    method: str = "rule_based",
) -> D.DiscriminatorResult:
    return D.DiscriminatorResult(
        p_oil=p_oil,
        p_class=p_class,
        method=method,  # type: ignore[arg-type]
        note="test",
        model_version=None,
        factors=[],
        unavailable={} if p_oil is not None else {"p_oil": D.INSUFFICIENT_FEATURES},
        base_p_oil=settings.rule_base_p_oil,
        evidence_fraction=1.0,
    )


# ------------------------------------------------------------------ blocking ----


def test_a_look_alike_may_not_name_a_culprit() -> None:
    """The exact contradiction that shipped: not oil, and yet a culprit."""
    decision = gate.evaluate(_result(0.481, D.CLASS_LOOK_ALIKE), wind_gate_violated=False)

    assert decision.passed is False
    assert decision.code == gate.BLOCKED_NOT_OIL
    assert "0.481" in decision.reason
    assert decision.recommendation


def test_oil_passes() -> None:
    decision = gate.evaluate(_result(0.895, D.CLASS_OIL), wind_gate_violated=False)

    assert decision.passed is True
    assert decision.code is None


def test_the_wind_gate_blocks_even_a_detection_that_reads_as_oil() -> None:
    """§5.2 is a validity condition on the observation, not a vote to be outweighed.

    A high P(oil) computed under a violated gate is a confident answer to a
    question that should not have been asked.
    """
    decision = gate.evaluate(
        _result(0.97, D.CLASS_OIL),
        wind_gate_violated=True,
        wind_gate_reason="wind 2.1 m/s is below the 3.0 m/s gate:",
    )

    assert decision.passed is False
    assert decision.code == gate.BLOCKED_WIND_GATE


def test_an_unscorable_detection_is_blocked_and_says_why() -> None:
    """No probability at all is not a low probability."""
    decision = gate.evaluate(_result(None, None), wind_gate_violated=False)

    assert decision.passed is False
    assert decision.code == gate.BLOCKED_UNSCORABLE
    assert decision.reason.strip()


def test_an_unmeasured_wind_gate_is_not_a_violated_one() -> None:
    """A gate that could not be evaluated must not be reported as failed.

    The classification still has to stand on its own — this only says the absence
    of wind is not itself the objection.
    """
    passing = gate.evaluate(_result(0.9, D.CLASS_OIL), wind_gate_violated=None)
    blocked = gate.evaluate(_result(0.2, D.CLASS_LOOK_ALIKE), wind_gate_violated=None)

    assert passing.passed is True
    assert blocked.code == gate.BLOCKED_NOT_OIL


def test_the_most_fundamental_objection_is_the_one_reported() -> None:
    """An unscorable detection under a violated gate reports unscorable.

    The reason goes on screen, so the order is not cosmetic: "could not be
    scored" and "scored, but the wind makes it untrustworthy" are different
    things to tell an officer.
    """
    decision = gate.evaluate(_result(None, None), wind_gate_violated=True)

    assert decision.code == gate.BLOCKED_UNSCORABLE


# ------------------------------------------------------- it can only withhold ----


def test_the_gate_never_promotes() -> None:
    """Exhaustive over the inputs: nothing but oil-and-gate-clear passes.

    This is the property the module exists for. If any other combination can
    return `passed`, the gate has become a second opinion rather than a filter.
    """
    classes = (D.CLASS_OIL, D.CLASS_LOOK_ALIKE, None)
    for p_class in classes:
        for violated in (True, False, None):
            p_oil = None if p_class is None else 0.9
            decision = gate.evaluate(_result(p_oil, p_class), wind_gate_violated=violated)
            expected = p_class == D.CLASS_OIL and not violated
            assert decision.passed is expected, (
                f"class={p_class} wind_gate_violated={violated} "
                f"passed={decision.passed}, expected {expected}"
            )


def test_the_gate_holds_the_threshold_it_is_told_not_its_own() -> None:
    """It re-uses `classify`'s verdict rather than re-deriving one.

    Two places deciding what counts as oil is one place too many — the detection
    just above the threshold must not be oil to the discriminator and look-alike
    to the gate.
    """
    threshold = settings.discriminator_oil_threshold
    just_over = D.classify(threshold + 1e-9)
    just_under = D.classify(threshold - 1e-9)

    assert gate.evaluate(_result(threshold + 1e-9, just_over), False).passed is True
    assert gate.evaluate(_result(threshold - 1e-9, just_under), False).passed is False


def test_a_passed_decision_has_no_refusal_to_render() -> None:
    decision = gate.evaluate(_result(0.9, D.CLASS_OIL), wind_gate_violated=False)

    with pytest.raises(ValueError):
        decision.as_refusal(n_vessels_in_frame=12)


def test_a_refusal_carries_no_candidates_and_an_unattributed_verdict() -> None:
    """§2.4 and §5.4: withholding is a first-class result, not an error path."""
    decision = gate.evaluate(_result(0.481, D.CLASS_LOOK_ALIKE), wind_gate_violated=False)
    refusal = decision.as_refusal(n_vessels_in_frame=23)

    assert refusal["issued"] is False
    assert refusal["verdict"] == "UNATTRIBUTED"
    assert refusal["candidates"] == []
    assert refusal["n_vessels_in_frame"] == 23
    assert refusal["blocked_by"] == gate.BLOCKED_NOT_OIL
    assert refusal["reason"] and refusal["recommendation"]


# ------------------------------------------------------ against the real scorer ----


def test_a_low_wind_look_alike_is_blocked_end_to_end() -> None:
    """SC-02's shape, through the real feature vector and the real scorer.

    Not a hand-built result: the gate has to block what the pipeline actually
    produces for that scene, not what a test says it produces.
    """
    features = F.ExtractedFeatures(
        values={  # type: ignore[typeddict-item]
            **dict.fromkeys(F.FEATURE_NAMES),
            "area_km2": 30.0,
            "shape_complexity": 1.2,
            "eccentricity": 0.30,
            "wind_speed_ms": 2.1,
        },
        unavailable={name: F.NO_PIXELS for name in F.RADIOMETRIC_FEATURES},
    )
    result = D.score(features)
    violated, reason = D.wind_gate(features)

    decision = gate.evaluate(result, violated, reason)

    assert result.p_class == D.CLASS_LOOK_ALIKE
    assert decision.passed is False
    assert decision.code == gate.BLOCKED_WIND_GATE
