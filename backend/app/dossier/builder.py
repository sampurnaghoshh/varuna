"""MARPOL Annex I evidence dossier — P2 (§10 item 14).

Not built in this build. The endpoint returns HTTP 501 (§7,
`app/api/v1/dossier.py`) and the frontend renders no control that reaches it.
Kept as a signature so the shape is settled if it is ever promoted.

Every number in a dossier would be injected from pipeline output, never
generated (§3).
"""

from pathlib import Path


def build_pdf(detection_id: int, mmsi: int, output_dir: Path) -> tuple[Path, str]:
    """Returns (pdf_path, sha256)."""
    ...
