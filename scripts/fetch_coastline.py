"""Fetch Natural Earth land and write the two Baltic land files the repo needs.

One download, two outputs, because two different jobs want land in two
incompatible shapes:

  data/geo/baltic_land.geojson       BASEMAP. Cartographic context for the
                                     offline console (§2.1) - MapLibre gets no
                                     tile server, so the map would otherwise
                                     have no land at all. Landmasses crossing
                                     the AOI edge are emitted as STROKED LINES,
                                     never as polygons, because a hand-rolled
                                     polygon clip bridges a concave coastline
                                     straight across open sea and fills it as
                                     land. A stroke has no fill to get wrong.

  data/geo/baltic_land_mask.geojson  LAND MASK. The authoritative "is this point
                                     on land" answer, used by
                                     scripts/land_guard.py. Real closed
                                     polygons, clipped by shapely, which does a
                                     correct polygon intersection and needs no
                                     bridging.

The basemap CANNOT serve as the mask, and the distinction is not cosmetic. In
the basemap only islands wholly inside the AOI survive as fills - eleven small
ones. Poland, Germany, Sweden and mainland Denmark all cross the AOI edge and
are therefore LineStrings. A point-in-polygon test against the basemap would
report a slick sitting on the Polish coast as open water, which is precisely the
failure the guard exists to catch.

The mask AOI is deliberately wider than the basemap's. A guard whose mask ends
before the drift does would read "off the edge of the mask" as "not on land", so
land_guard.py refuses geometry that leaves the declared AOI rather than assuming
the sea continues.

Human-invoked. Needs network and shapely, so run it in the backend container:

    docker compose exec -T backend python scripts/fetch_coastline.py

Nothing under §9's read-only paths is touched - data/geo/ is neither
data/scenes/ nor data/ais/.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)

# Baltic AOI — comfortably wider than the three scene footprints so the map can
# be panned out without running off the edge of the land data.
LON_MIN, LON_MAX = 13.5, 17.5
LAT_MIN, LAT_MAX = 54.0, 56.5

# Mask AOI — wider still. The guard checks 12 h particle clouds, which travel
# tens of kilometres from the scene, and a mask that stopped at the basemap edge
# would silently score anything beyond it as water.
MASK_LON_MIN, MASK_LON_MAX = 13.0, 18.0
MASK_LAT_MIN, MASK_LAT_MAX = 53.5, 57.0

REPO_ROOT = Path(__file__).resolve().parent.parent


def _geo_dir() -> Path:
    """Where the two files land, on the host and in the container alike.

    In the container only ./backend and ./scripts are mounted, so walking up
    from __file__ lands in /app and writes into the backend bind mount - the
    files appear under backend/data/geo/ and the repo's real data/geo/ stays
    stale, with nothing to say it happened. VARUNA_GEO_DIR is the container's
    answer (compose sets it to the mounted /data/geo); the repo-root path is the
    host's.
    """
    override = os.environ.get("VARUNA_GEO_DIR")
    return Path(override) if override else REPO_ROOT / "data" / "geo"


OUT_NAME = "baltic_land.geojson"
MASK_NAME = "baltic_land_mask.geojson"

Ring = list[list[float]]


def _in_aoi(pt: list[float]) -> bool:
    return LON_MIN <= pt[0] <= LON_MAX and LAT_MIN <= pt[1] <= LAT_MAX


def _bbox(ring: Iterable[list[float]]) -> tuple[float, float, float, float]:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return min(lons), max(lons), min(lats), max(lats)


def _bbox_hits(ring: Iterable[list[float]]) -> bool:
    lo_x, hi_x, lo_y, hi_y = _bbox(ring)
    return lo_x <= LON_MAX and hi_x >= LON_MIN and lo_y <= LAT_MAX and hi_y >= LAT_MIN


def _wholly_inside(ring: Iterable[list[float]]) -> bool:
    lo_x, hi_x, lo_y, hi_y = _bbox(ring)
    return LON_MIN <= lo_x and hi_x <= LON_MAX and LAT_MIN <= lo_y and hi_y <= LAT_MAX


def _clip_segment(
    a: list[float], b: list[float]
) -> tuple[list[float], list[float]] | None:
    """Liang-Barsky clip of one segment against the AOI rectangle."""
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - LON_MIN),
        (dx, LON_MAX - x1),
        (-dy, y1 - LAT_MIN),
        (dy, LAT_MAX - y1),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return (
        [x1 + t0 * dx, y1 + t0 * dy],
        [x1 + t1 * dx, y1 + t1 * dy],
    )


def clip_ring_to_lines(ring: Ring) -> list[Ring]:
    """Clip a closed ring to the AOI as open polylines.

    Deliberately NOT a polygon clip. Sutherland-Hodgman closes a concave
    subject by bridging along the clip border, and for a coastline that bridge
    runs straight across open sea and fills it as land. A coastline rendered as
    a stroked line has no fill to get wrong, so this returns line segments and
    lets the caller stroke them.
    """
    lines: list[Ring] = []
    current: Ring = []
    for i in range(len(ring) - 1):
        clipped = _clip_segment(ring[i], ring[i + 1])
        if clipped is None:
            if len(current) >= 2:
                lines.append(current)
            current = []
            continue
        start, end = clipped
        if not current:
            current = [start, end]
        elif (
            abs(current[-1][0] - start[0]) < 1e-9
            and abs(current[-1][1] - start[1]) < 1e-9
        ):
            current.append(end)
        else:
            if len(current) >= 2:
                lines.append(current)
            current = [start, end]
    if len(current) >= 2:
        lines.append(current)
    return lines


def _round(ring: Ring) -> Ring:
    return [[round(p[0], 5), round(p[1], 5)] for p in ring]


def build_land_mask(source: dict[str, Any]) -> dict[str, Any]:
    """A true polygon land mask over the mask AOI, for scripts/land_guard.py.

    shapely's intersection is a real polygon clip, so unlike the basemap path
    above this can keep closed rings without bridging a coastline across open
    sea. `buffer(0)` repairs the handful of self-intersecting rings Natural
    Earth ships; without it unary_union raises on them.

    The AOI travels inside the file. The guard reads it back and refuses any
    geometry that leaves it, so "outside the mask" can never be mistaken for
    "not on land".
    """
    from shapely.geometry import box, mapping, shape
    from shapely.ops import unary_union

    aoi = box(MASK_LON_MIN, MASK_LAT_MIN, MASK_LON_MAX, MASK_LAT_MAX)
    parts = []
    for feature in source["features"]:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        land = shape(geometry)
        if not land.is_valid:
            land = land.buffer(0)
        if land.intersects(aoi):
            parts.append(land.intersection(aoi))

    merged = unary_union(parts)
    return {
        "type": "Feature",
        "properties": {
            "source": "Natural Earth 1:10m land (public domain)",
            "source_url": SOURCE_URL,
            "aoi": {
                "lon": [MASK_LON_MIN, MASK_LON_MAX],
                "lat": [MASK_LAT_MIN, MASK_LAT_MAX],
            },
            "note": (
                "Authoritative land mask for scripts/land_guard.py. Closed "
                "polygons from a shapely intersection, not the stroked basemap "
                "in baltic_land.geojson - that one drops every landmass "
                "crossing the AOI edge to a LineString and cannot answer a "
                "point-in-polygon question."
            ),
            "aoi_note": (
                "Geometry outside this AOI is not water; it is unknown. The "
                "guard raises rather than assuming."
            ),
        },
        "geometry": mapping(merged),
    }


def main() -> None:
    print(f"Downloading {SOURCE_URL} ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=180) as response:
        source = json.loads(response.read().decode("utf-8"))
    print(f"  {len(source['features'])} source features")

    features: list[dict[str, object]] = []
    n_filled = 0
    n_stroked = 0

    for feature in source["features"]:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Polygon":
            candidates = [geometry["coordinates"]]
        elif geometry.get("type") == "MultiPolygon":
            candidates = geometry["coordinates"]
        else:
            continue
        for rings in candidates:
            if not rings or not _bbox_hits(rings[0]):
                continue
            if _wholly_inside(rings[0]):
                # An island entirely within the AOI needs no clipping, so it
                # can be filled with no risk of a bridging artefact.
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"render": "fill"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [_round(r) for r in rings],
                        },
                    }
                )
                n_filled += 1
                continue
            for ring in rings:
                for line in clip_ring_to_lines(ring):
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {"render": "stroke"},
                            "geometry": {
                                "type": "LineString",
                                "coordinates": _round(line),
                            },
                        }
                    )
                    n_stroked += 1

    out = {
        "type": "FeatureCollection",
        "properties": {
            "source": "Natural Earth 1:10m land (public domain)",
            "source_url": SOURCE_URL,
            "clipped_to": {"lon": [LON_MIN, LON_MAX], "lat": [LAT_MIN, LAT_MAX]},
            "render_note": (
                "properties.render is 'fill' for islands wholly inside the AOI "
                "and 'stroke' for coastline clipped at the AOI edge. Landmasses "
                "crossing the edge are lines, not polygons, so no fill can bleed "
                "across open sea."
            ),
            "note": (
                "Basemap land for the offline console. Cartographic context "
                "only - nothing in the pipeline reads this file."
            ),
        },
        "features": features,
    }

    geo_dir = _geo_dir()
    geo_dir.mkdir(parents=True, exist_ok=True)

    out_path = geo_dir / OUT_NAME
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path}")
    print(f"  {n_filled} filled islands, {n_stroked} coastline lines, {size_kb:.1f} KB")

    mask_path = geo_dir / MASK_NAME
    mask = build_land_mask(source)
    mask_path.write_text(json.dumps(mask, separators=(",", ":")), encoding="utf-8")
    mask_kb = mask_path.stat().st_size / 1024
    print(f"Wrote {mask_path}")
    print(f"  land mask over {mask['properties']['aoi']}, {mask_kb:.1f} KB")


if __name__ == "__main__":
    main()
