"""Forward-impact endpoint — §7, P2 (§10 item 15).

    GET /impact/{detection_id}

Kept in the contract, not implemented in this build. Returns HTTP 501 with a
typed message (§7). The frontend renders no control that reaches it.
"""

from fastapi import APIRouter, HTTPException, status

router = APIRouter()

_NOT_IMPLEMENTED = "Forward impact is P2 and not implemented in this build."


@router.get("/impact/{detection_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_impact(detection_id: int) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_NOT_IMPLEMENTED,
    )
