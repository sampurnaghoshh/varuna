"""CFAR / adaptive-threshold dark-formation detector — the always-works fallback.

§9: keep this dependency-light and never let it import torch. It is the reason
the demo survives a missing weights file, a broken CUDA install, or a laptop
that cannot load the U-Net. numpy and scipy only.
"""

import numpy as np
from shapely.geometry import Polygon


def adaptive_threshold(
    sigma0_db: np.ndarray,
    window_px: int,
    offset_db: float,
) -> np.ndarray:
    """Local-mean CFAR: flag pixels sitting `offset_db` below the local mean."""
    ...


def detect(
    sigma0_db: np.ndarray,
    transform: object,
    min_area_km2: float,
) -> list[Polygon]: ...
