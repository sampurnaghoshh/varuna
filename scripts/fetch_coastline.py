"""Fetch and clip a Baltic land polygon for the offline console basemap.

The demo runs with no internet (§2.1), so MapLibre gets no tile server and the
map would otherwise have no land at all. This bakes a coarse coastline into the
repo instead: Natural Earth 1:10m land, clipped to the Baltic AOI.

Human-invoked, once. Writes data/geo/baltic_land.geojson. Nothing under §9's
read-only paths is touched — data/geo/ is a new directory.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_land.geojson"
)

# Baltic AOI — comfortably wider than the three scene footprints so the map can
# be panned out without running off the edge of the land data.
LON_MIN, LON_MAX = 13.5, 17.5
LAT_MIN, LAT_MAX = 54.0, 56.5

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "geo" / "baltic_land.geojson"

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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH}")
    print(f"  {n_filled} filled islands, {n_stroked} coastline lines, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
