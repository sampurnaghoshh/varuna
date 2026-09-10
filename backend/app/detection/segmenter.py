"""U-Net inference over a Sentinel-1 Sigma0 scene — §10 P0-2.

Weights are trained on Part III indices 001-120 and validated on 121-145; the
holdout range 146-150 is never seen (§0, §9). Training lives in
`ml/train_segmenter.py`, which imports the split from `scripts/data_split.py`.

Degrades to `classical.py` when weights or torch are unavailable (§8).
"""

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def load_model(weights_path: Path): ...


def segment(scene_array: np.ndarray) -> np.ndarray:
    """Return a binary dark-formation mask for a (2, H, W) VV+VH dB array."""
    ...


def mask_to_polygons(
    mask: np.ndarray,
    transform: object,
    min_area_km2: float,
) -> list[Polygon]: ...
