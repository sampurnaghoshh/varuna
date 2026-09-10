"""Particle cloud to probability field and uncertainty hull — §5.1.

The KDE probability field O(x, y, t) is normalised to sum to 1 across the WHOLE
run, not per snapshot: attribution channel E1 integrates mass over time, so
per-snapshot normalisation would weight every snapshot equally regardless of how
concentrated the cloud was.

Density is computed here, not in solver.py: the solver produces kinematics, this
module interprets them. The grid is built once for the whole run - E1 sums mass
across snapshots, and per-snapshot grids cannot be summed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import shapely
import shapely.prepared
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Polygon
from shapely.ops import polygonize

from app.config import settings

logger = logging.getLogger(__name__)


class SnapshotLike(Protocol):
    """What density needs from a solver snapshot - nothing more.

    Structural, not an import of DriftState: density interprets clouds, and
    keeping the dependency one-directional means the solver can never grow a
    circular import back into here.
    """

    t_offset_min: int
    positions: np.ndarray
    stretch_factor: float
    stretch_factor_raw: float


@dataclass(frozen=True)
class DensityGrid:
    """Run-level raster in the solver's UTM frame. Shared by every snapshot."""

    x0_m: float
    y0_m: float
    cell_size_m: float
    nx: int
    ny: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0_m + (np.arange(self.nx) + 0.5) * self.cell_size_m
        y = self.y0_m + (np.arange(self.ny) + 0.5) * self.cell_size_m
        return x, y


@dataclass(frozen=True)
class SnapshotDensity:
    t_offset_min: int
    probability: np.ndarray
    hull: Polygon
    stretch_factor: float
    stretch_factor_raw: float


@dataclass(frozen=True)
class RunDensity:
    grid: DensityGrid
    snapshots: list[SnapshotDensity]

    def total_mass(self) -> float:
        return float(sum(s.probability.sum() for s in self.snapshots))


def build_grid(
    all_positions: list[np.ndarray],
    cell_size_m: float | None = None,
    pad_m: float | None = None,
) -> DensityGrid:
    """One grid covering every snapshot of the run, plus a pad for the kernel."""
    cell_size_m = settings.drift_cell_size_m if cell_size_m is None else cell_size_m
    pad_m = settings.drift_grid_pad_m if pad_m is None else pad_m

    stacked = np.concatenate([p for p in all_positions if p.size], axis=0)
    if stacked.size == 0:
        raise ValueError("Cannot build a density grid from zero particles")

    x_min = float(stacked[:, 0].min()) - pad_m
    y_min = float(stacked[:, 1].min()) - pad_m
    x_max = float(stacked[:, 0].max()) + pad_m
    y_max = float(stacked[:, 1].max()) + pad_m

    nx = max(int(np.ceil((x_max - x_min) / cell_size_m)), 1)
    ny = max(int(np.ceil((y_max - y_min) / cell_size_m)), 1)
    return DensityGrid(x0_m=x_min, y0_m=y_min, cell_size_m=cell_size_m, nx=nx, ny=ny)


def kde_field(
    positions: np.ndarray,
    grid: DensityGrid,
    bandwidth_m: float | None = None,
) -> np.ndarray:
    """Gaussian KDE on the run grid: 2-D histogram convolved with a Gaussian.

    Mathematically the same estimator as evaluating a Gaussian kernel at every
    cell, but O(N + G) instead of O(N * G). At 5000 particles x 25 snapshots x a
    grid this size the direct form costs minutes and would blow the §7 live-stage
    budget; this costs milliseconds. It is also exactly reproducible, which the
    offline demo depends on (§2.1).

    Returns unnormalised density. Run-wide normalisation is `normalise_run`.
    """
    bandwidth_m = settings.drift_kde_bandwidth_m if bandwidth_m is None else bandwidth_m

    counts, _, _ = np.histogram2d(
        positions[:, 1],
        positions[:, 0],
        bins=(grid.ny, grid.nx),
        range=(
            (grid.y0_m, grid.y0_m + grid.ny * grid.cell_size_m),
            (grid.x0_m, grid.x0_m + grid.nx * grid.cell_size_m),
        ),
    )
    sigma_cells = bandwidth_m / grid.cell_size_m
    return gaussian_filter(counts, sigma=sigma_cells, mode="constant", cval=0.0)


def normalise_run(fields: list[np.ndarray]) -> list[np.ndarray]:
    """Normalise so the sum over every cell of every snapshot is 1.0."""
    total = float(sum(float(f.sum()) for f in fields))
    if total <= 0.0:
        raise ValueError("Cannot normalise a run whose total density is zero")
    return [f / total for f in fields]


def auto_alpha(positions: np.ndarray) -> float:
    """Alpha (1/metres) derived from the cloud's own particle spacing.

    A hand-picked alpha is a magic number that silently becomes wrong the moment
    the particle count or the horizon changes. Scaling the circumradius cutoff to
    a multiple of the mean inter-particle spacing keeps the hull tight on a dense
    cloud and loose on a sparse one, deterministically.
    """
    if positions.shape[0] < 4:
        return 0.0
    spread = float(np.sqrt(np.prod(positions.std(axis=0) + 1e-9)))
    spacing = spread / np.sqrt(positions.shape[0])
    cutoff_m = max(settings.drift_hull_spacing_multiple * spacing, 1.0)
    return 1.0 / cutoff_m


def alpha_hull(positions: np.ndarray, alpha: float | None = None) -> Polygon:
    """Alpha shape of the particle cloud: the uncertainty polygon (§5.1).

    Delaunay triangles whose circumradius exceeds 1/alpha are dropped, so the
    boundary follows concavities a convex hull would bridge over. That matters
    here: a convex hull across a horseshoe-shaped rewound cloud claims origin
    probability in the gap, where no particle ever went.

    Built by polygonizing the boundary edges - those belonging to exactly one
    kept triangle - rather than unioning the triangles themselves. Both give the
    same shape, but the union of ~9000 triangles costs 0.8 s per snapshot, which
    is 19 s across a run and blows the §7 live-stage budget on its own.

    Degrades to the convex hull when the alpha shape collapses or fragments -
    an over-tight alpha must not return an empty origin polygon.
    """
    unique = np.unique(positions, axis=0)

    def convex() -> Polygon:
        # Built vectorised and only when actually needed. Constructing it eagerly
        # as a fallback costs a Python shapely Point per particle - 3.3 s across a
        # run, most of this module's budget, for a branch almost never taken.
        return shapely.convex_hull(shapely.multipoints(unique))

    if unique.shape[0] < 4:
        return convex()

    alpha = auto_alpha(unique) if alpha is None else alpha
    if alpha <= 0.0:
        return convex()

    try:
        triangulation = Delaunay(unique)
    except QhullError:
        logger.warning("drift.density: Delaunay failed; falling back to convex hull")
        return convex()

    simplices = unique[triangulation.simplices]
    a = np.linalg.norm(simplices[:, 0] - simplices[:, 1], axis=1)
    b = np.linalg.norm(simplices[:, 1] - simplices[:, 2], axis=1)
    c = np.linalg.norm(simplices[:, 2] - simplices[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        circumradius = np.where(area > 0.0, (a * b * c) / (4.0 * area), np.inf)

    keep = circumradius < (1.0 / alpha)
    if not keep.any():
        return convex()

    kept_tris = triangulation.simplices[keep]
    edges = np.sort(
        np.concatenate([kept_tris[:, [0, 1]], kept_tris[:, [1, 2]], kept_tris[:, [2, 0]]]),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if boundary.size == 0:
        return convex()

    # Vectorised construction: one C call for every boundary edge rather than a
    # Python LineString per edge, which is most of this function's cost.
    endpoints = unique[boundary.reshape(-1)]
    lines = shapely.linestrings(
        endpoints, indices=np.repeat(np.arange(boundary.shape[0]), 2)
    )
    rings = list(polygonize(lines))
    if not rings:
        return convex()

    outer = max(rings, key=lambda g: g.area)
    prepared_outer = shapely.prepared.prep(outer)
    holes = [
        ring.exterior.coords
        for ring in rings
        if ring is not outer and prepared_outer.contains(ring.representative_point())
    ]
    hull = Polygon(outer.exterior.coords, holes)
    if not hull.is_valid:
        hull = hull.buffer(0)
    if hull.is_empty or not isinstance(hull, Polygon):
        return convex()
    return hull


def compute_run_density(
    states: Sequence[SnapshotLike],
    cell_size_m: float | None = None,
    bandwidth_m: float | None = None,
    alpha: float | None = None,
) -> RunDensity:
    """Grid, per-snapshot KDE and hull for a whole DriftRun.

    Takes states rather than the run itself so it stays usable on any sequence of
    snapshots and keeps this module free of a solver import.
    """
    if not states:
        raise ValueError("Cannot compute density for a run with no states")

    grid = build_grid([s.positions for s in states], cell_size_m=cell_size_m)
    raw = [kde_field(s.positions, grid, bandwidth_m=bandwidth_m) for s in states]
    probability = normalise_run(raw)

    snapshots = [
        SnapshotDensity(
            t_offset_min=state.t_offset_min,
            probability=prob,
            hull=alpha_hull(state.positions, alpha=alpha),
            stretch_factor=state.stretch_factor,
            stretch_factor_raw=state.stretch_factor_raw,
        )
        for state, prob in zip(states, probability, strict=True)
    ]
    logger.info(
        "drift.density: %d snapshots on a %dx%d grid at %.0f m",
        len(snapshots),
        grid.nx,
        grid.ny,
        grid.cell_size_m,
    )
    return RunDensity(grid=grid, snapshots=snapshots)
