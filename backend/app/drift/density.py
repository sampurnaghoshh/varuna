"""Particle cloud to probability field and uncertainty hull — §5.1.

The KDE probability field O(x, y, t) is normalised to sum to 1 across the WHOLE
run, not per snapshot: attribution channel E1 integrates mass over time, so
per-snapshot normalisation would weight every snapshot equally regardless of how
concentrated the cloud was.
"""

import numpy as np
from shapely.geometry import Polygon


def alpha_hull(positions: np.ndarray, alpha: float) -> Polygon: ...


def kde_field(
    positions: np.ndarray,
    bbox: tuple[float, float, float, float],
    cell_size_m: float,
    bandwidth_m: float,
) -> np.ndarray: ...


def normalise_run(fields: list[np.ndarray]) -> list[np.ndarray]:
    """Normalise so the sum over every cell of every snapshot is 1.0."""
    ...
