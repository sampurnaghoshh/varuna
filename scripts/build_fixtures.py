"""Capture real engine output into data/fixtures/ (`make fixtures`).

HUMAN-INVOKED ONLY (§13). Per §15 this runs at hour 5, once the drift solver and
the attribution engine can run end-to-end on all three Part III demo scenes -
well enough to emit real output, not polished.

It writes, per scenario: detections + features + SHAP factors, drift frames at
30-min steps, ranked attributions with channel breakdowns, and the full WS event
sequence.

Every number captured here is a genuine engine output. Nothing in this file
fabricates a value - §2.2 applies here above all, because these fixtures are
what the demo falls back to and what the frontend is built against.

Once green the fixtures are FROZEN (§9). If a fixture is genuinely wrong, say so
and stop - a human decides.
"""

import sys


def build_scenario(scenario_code: str) -> None: ...


def main() -> int:
    print("[build_fixtures] phase 1 stub - fixtures are generated at hour 5 (§15)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
