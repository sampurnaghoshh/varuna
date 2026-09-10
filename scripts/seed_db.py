"""Seed the database: schema, scenes, AIS, demo scenarios (`make seed`).

Phase 1 stub. The schema itself is applied by the postgis container from
backend/app/db/schema.sql on an empty volume; this script loads the three
holdout demo scenes (§12), the Danish AIS subset, and the demo_scenarios rows.

data/scenes/** and data/ais/** are read-only inputs and are never modified in
place (§9). Scene ingest is deliberately split-unaware (§0).
"""

import sys


def seed_scenes() -> int: ...


def seed_ais() -> int: ...


def seed_demo_scenarios() -> int: ...


def main() -> int:
    print("[seed_db] phase 1 stub - schema is applied by the db container on first up")
    print("[seed_db] scenes / AIS / demo scenarios land with their owning modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
