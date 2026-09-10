"""Fixture loader — serves data/fixtures/ when the live path fails (§10 P0-9).

Per §15 this is not a late bolt-on: from hour 5 the frontend is developed
entirely against these fixtures, so "source": "fixture" (§7) is the normal
development case, not an error state.

Fixtures are FROZEN once green (§9). This module reads them and never writes.

It also imports nothing from app.db, sqlalchemy, asyncpg or redis, and it must
not start to: the offline demo (§2.1) has to replay with the database stopped,
and a single transitive import of the data layer would silently take that away.
`backend/tests/test_fixtures.py` asserts it on the import graph rather than
trusting the convention.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from app.config import settings

logger = logging.getLogger(__name__)

Source = Literal["live", "fixture"]

MANIFEST = "manifest.json"

# Names are used to build filesystem paths, so they are matched rather than
# sanitised: an artefact key is a file stem chosen by build_fixtures.py, and
# anything that is not one is a bug in the caller, not input to be repaired.
_SCENARIO_RE = re.compile(r"^SC-\d{2}$")
_ARTEFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FixtureNotFoundError(FileNotFoundError):
    """Raised for a fixture that is not on disk.

    Callers gate on `has_fixture` and degrade to a typed error (§8); this is
    never surfaced to the user.
    """


class MalformedFixtureNameError(ValueError):
    """Raised for a scenario or artefact name that cannot address a fixture."""


def fixtures_dir() -> Path:
    return Path(settings.fixtures_dir)


def _scenario_dir(scenario_code: str) -> Path:
    if not _SCENARIO_RE.match(scenario_code):
        raise MalformedFixtureNameError(
            f"Scenario code {scenario_code!r} is not of the form SC-01."
        )
    return fixtures_dir() / scenario_code


def _artefact_path(scenario_code: str, artefact: str) -> Path:
    if not _ARTEFACT_RE.match(artefact) or ".." in artefact:
        raise MalformedFixtureNameError(f"Artefact name {artefact!r} is not a fixture key.")
    return _scenario_dir(scenario_code) / f"{artefact}.json"


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise FixtureNotFoundError(f"No fixture at {path}") from exc


def manifest() -> dict[str, Any]:
    """The fixture set's own account of itself — provenance, badges, what is pending.

    The UI reads its badges from here. A provisional set that does not say so is
    the failure §2.2 exists to prevent, so this is not optional metadata.
    """
    return _read_json(fixtures_dir() / MANIFEST)


def scenarios() -> list[str]:
    """Scenario codes present on disk, in manifest order where one exists."""
    try:
        declared = [str(entry["code"]) for entry in manifest().get("scenarios", [])]
    except FixtureNotFoundError:
        declared = []
    present = {p.name for p in fixtures_dir().glob("SC-*") if p.is_dir()}
    ordered = [code for code in declared if code in present]
    return ordered + sorted(present - set(ordered))


def artefacts(scenario_code: str) -> list[str]:
    """Artefact keys available for a scenario — file stems, sorted."""
    directory = _scenario_dir(scenario_code)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def has_fixture(scenario_code: str, artefact: str) -> bool:
    try:
        return _artefact_path(scenario_code, artefact).is_file()
    except MalformedFixtureNameError:
        return False


def load(scenario_code: str, artefact: str) -> dict[str, Any]:
    """One artefact, parsed. Raises FixtureNotFoundError when it is absent.

    Deliberately uncached. These files are read a handful of times per demo run
    and a cache would either hand callers a shared mutable dict or cost a deep
    copy per read; neither is worth it, and re-reading keeps the fixtures
    genuinely read-only (§9).
    """
    payload = _read_json(_artefact_path(scenario_code, artefact))
    if not isinstance(payload, dict):
        raise FixtureNotFoundError(
            f"Fixture {scenario_code}/{artefact} is not an object; got {type(payload).__name__}"
        )
    return payload


def load_ws_sequence(scenario_code: str) -> list[dict[str, Any]]:
    """The scenario's full §7 PipelineEvent sequence, in order.

    Each event carries `playback_offset_ms` — the authored pacing the demo
    controller sleeps on. Measured engine time lives once per scenario under the
    file's `timings_ms`, deliberately apart from the pacing: one is a measurement
    and the other is a presentation value, and the UI must never render the
    second as the first (§2.2).
    """
    events = load(scenario_code, "ws").get("events", [])
    return [dict(event) for event in events]


def envelope(payload: dict[str, Any], elapsed_ms: int) -> dict[str, Any]:
    """Wrap a fixture payload in the §7 response envelope.

    `source` is pinned to "fixture" here and nowhere else: a fixture-backed
    answer is never presented as a live one (§2.2).

    The envelope is stamped AFTER the payload is spread, not before. A payload
    carrying its own "source" - a scene's provenance, say - would otherwise
    overwrite the stamp and the response would claim to be live.
    """
    return {**payload, "source": "fixture", "elapsed_ms": int(elapsed_ms)}
