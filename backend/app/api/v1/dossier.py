"""Dossier endpoint — §7, P2 (§10 item 14).

    POST /dossier/{detection_id}/{mmsi}

Kept in the contract, not implemented in this build. It returns HTTP 501 with a
typed message — never a 404, never an empty 200, never a plausible-looking stub
payload (§2.2). The frontend renders no control that reaches it (§7, no dead
buttons).
"""

from fastapi import APIRouter, HTTPException, status

router = APIRouter()

_NOT_IMPLEMENTED = "Evidence dossier is P2 and not implemented in this build."


@router.post("/dossier/{detection_id}/{mmsi}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def build_dossier(detection_id: int, mmsi: int) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_NOT_IMPLEMENTED,
    )
