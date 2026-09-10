"""Fixture loader — serves data/fixtures/ when the live path fails (§10 P0-9).

Per §15 this is not a late bolt-on: from hour 5 the frontend is developed
entirely against these fixtures, so "source": "fixture" (§7) is the normal
development case, not an error state.

Fixtures are FROZEN once green (§9). This module reads them and never writes.
"""

from typing import Any, Literal

Source = Literal["live", "fixture"]


def has_fixture(scenario_code: str, artefact: str) -> bool: ...


def load(scenario_code: str, artefact: str) -> dict[str, Any]: ...


def load_ws_sequence(scenario_code: str) -> list[dict[str, Any]]: ...
