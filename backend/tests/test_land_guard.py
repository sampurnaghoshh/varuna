"""The placement guard — `scripts/land_guard.py`.

The bug this exists to prevent already happened once: SC-01 was generated over
Bornholm with 80 of 116 slick vertices, the release point and ~90% of the
particles ashore, and every number downstream was arithmetically correct and
physically meaningless. Nothing failed, because nothing had an opinion about
land.

So the tests that matter are the ones about *raising*. A guard that reports a
problem without stopping the build is the state we were already in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, mapping, shape

from scripts import land_guard

# Bornholm. The island SC-01 was originally generated on top of.
BORNHOLM_LON, BORNHOLM_LAT = 14.90, 55.15

# Eastern Bornholm Basin — where SC-01 actually lives now, 53 km off the nearest
# land. Kept here as a literal rather than imported from build_fixtures so that
# moving the scenario cannot silently move the test's idea of open water.
OPEN_WATER_LON, OPEN_WATER_LAT = 16.30, 55.55


@pytest.fixture(scope="module")
def mask() -> land_guard.LandMask:
    try:
        return land_guard.load_mask()
    except land_guard.LandMaskUnavailableError as error:
        pytest.skip(str(error))


# ------------------------------------------------------------------- the mask ----


def test_the_mask_is_polygons_not_the_stroked_basemap(mask: land_guard.LandMask) -> None:
    """The basemap cannot answer a point-in-polygon question, and must not be used.

    In `baltic_land.geojson` every landmass crossing the AOI edge — Poland,
    Germany, Sweden, mainland Denmark — is a LineString, so only eleven small
    islands are fillable. A guard reading that file would report the Polish coast
    as open water.
    """
    assert mask.land.area > 0.0
    assert mask.on_land(np.array([BORNHOLM_LON]), np.array([BORNHOLM_LAT]))[0]


def test_the_mask_covers_mainland_the_basemap_drops(mask: land_guard.LandMask) -> None:
    """Coastlines clipped to lines in the basemap are real polygons here.

    Points well inside Poland, Germany and Sweden. If any of these reads as water
    the mask has silently degraded to the basemap's coverage and the guard is
    checking almost nothing.
    """
    for name, lon, lat in (
        ("Poland", 16.50, 54.30),
        ("Germany", 13.60, 54.00),
        ("Sweden", 14.50, 56.20),
    ):
        assert mask.on_land(np.array([lon]), np.array([lat]))[0], f"{name} read as water"


def test_open_water_is_open_water(mask: land_guard.LandMask) -> None:
    assert not mask.on_land(np.array([OPEN_WATER_LON]), np.array([OPEN_WATER_LAT]))[0]


# ---------------------------------------------------------------- the raising ----


def test_a_land_placed_polygon_raises(mask: land_guard.LandMask) -> None:
    """The deliberate land case: a slick drawn across Bornholm."""
    on_bornholm = Point(BORNHOLM_LON, BORNHOLM_LAT).buffer(0.03)

    with pytest.raises(land_guard.LandPlacementError) as error:
        land_guard.check_polygon(mask, "deliberate land case", on_bornholm)

    assert "on land" in str(error.value)


def test_a_land_placed_release_point_raises(mask: land_guard.LandMask) -> None:
    with pytest.raises(land_guard.LandPlacementError):
        land_guard.check_points(
            mask,
            "release point",
            np.array([BORNHOLM_LON]),
            np.array([BORNHOLM_LAT]),
        )


def test_an_open_water_polygon_passes_and_reports_its_clearance(
    mask: land_guard.LandMask,
) -> None:
    patch = Point(OPEN_WATER_LON, OPEN_WATER_LAT).buffer(0.05)
    report = land_guard.check_polygon(mask, "open water", patch)

    assert report.n_on_land == 0
    assert report.fraction_on_land == 0.0
    assert report.clearance_km is not None and report.clearance_km > 10.0


# --- the particle threshold ---------------------------------------------------
# A 12 h cloud is allowed tails; it is not allowed to be mostly ashore.


def _cloud(fraction_on_land: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A cloud with a known fraction of its particles on Bornholm."""
    n_land = int(round(fraction_on_land * n))
    lon = np.concatenate(
        (np.full(n_land, BORNHOLM_LON), np.full(n - n_land, OPEN_WATER_LON))
    )
    lat = np.concatenate(
        (np.full(n_land, BORNHOLM_LAT), np.full(n - n_land, OPEN_WATER_LAT))
    )
    return lon, lat


def test_a_cloud_over_the_tolerance_raises(mask: land_guard.LandMask) -> None:
    lon, lat = _cloud(0.06)

    with pytest.raises(land_guard.LandPlacementError) as error:
        land_guard.check_points(
            mask,
            "particles",
            lon,
            lat,
            max_fraction_on_land=land_guard.MAX_PARTICLE_LAND_FRACTION,
        )

    assert "6.0%" in str(error.value)


def test_a_cloud_under_the_tolerance_passes(mask: land_guard.LandMask) -> None:
    """Tails are physical: a few particles grounding is a statement, not a bug."""
    lon, lat = _cloud(0.04)
    report = land_guard.check_points(
        mask,
        "particles",
        lon,
        lat,
        max_fraction_on_land=land_guard.MAX_PARTICLE_LAND_FRACTION,
    )
    assert report.fraction_on_land == pytest.approx(0.04)


def test_the_tolerance_does_not_apply_to_slicks_or_release_points(
    mask: land_guard.LandMask,
) -> None:
    """One vertex ashore is enough. The particle allowance is not a global one."""
    lon, lat = _cloud(0.01, n=100)

    with pytest.raises(land_guard.LandPlacementError):
        land_guard.check_points(mask, "slick vertices", lon, lat)


# ---------------------------------------------------------------- the AOI edge ----


def test_geometry_outside_the_mask_raises_rather_than_reading_as_water(
    mask: land_guard.LandMask,
) -> None:
    """"Off the edge of the mask" is not "not on land".

    The North Sea is water, but this mask cannot say so, and a guard that
    answered anyway would pass every placement outside the Baltic — including
    ones on land.
    """
    with pytest.raises(land_guard.LandPlacementError) as error:
        land_guard.check_points(mask, "far west", np.array([2.0]), np.array([56.0]))

    assert "outside the land mask AOI" in str(error.value)


def test_an_empty_point_set_raises_rather_than_passing_by_default(
    mask: land_guard.LandMask,
) -> None:
    """Nothing to check is not the same as nothing wrong."""
    with pytest.raises(land_guard.LandPlacementError):
        land_guard.check_points(mask, "empty", np.array([]), np.array([]))


# --------------------------------------------------------------- unavailability ----


def test_a_missing_mask_is_fatal_not_skipped(tmp_path) -> None:
    """A guard that disables itself when its reference data is absent is not a guard."""
    with pytest.raises(land_guard.LandMaskUnavailableError):
        land_guard.load_mask(tmp_path / "nope.geojson")


def test_a_mask_without_a_declared_aoi_is_refused(tmp_path) -> None:
    """Without an AOI the guard could not tell water from off-the-edge."""
    path = tmp_path / "no_aoi.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(land_guard.LandMaskUnavailableError):
        land_guard.load_mask(path)


# --------------------------------------------------------- the shipped fixtures ----


def test_the_shipped_sc01_slick_is_over_water(mask: land_guard.LandMask) -> None:
    """The regression, asserted against what is actually committed.

    Not a re-run of the generator — the point is that the fixture ON DISK, the
    one the offline demo replays, has no oil on land.
    """
    from app.fallback import loader

    if not loader.has_fixture("SC-01", "detection"):
        pytest.skip("no fixtures on disk; run `make fixtures` first")

    slick = shape(loader.load("SC-01", "detection")["geom"])
    report = land_guard.check_polygon(mask, "shipped SC-01 slick", slick)

    assert report.n_on_land == 0
    assert report.clearance_km is not None and report.clearance_km > 5.0


# ------------------------------------------------- the generator, end to end ----


def test_the_generator_refuses_to_build_a_land_placed_scenario(tmp_path) -> None:
    """The deliberate land case, through `build_fixtures.build_sc01` itself.

    The unit tests above prove the guard can raise. This proves it is actually
    wired into the path that produced the bug — SC-01 re-pointed at Bornholm,
    exactly where it used to sit, and the build stops.

    It raises before the first solver call, so this costs nothing, and it writes
    nothing: no directory is created and the shipped fixtures are untouched.
    """
    from scripts import build_fixtures

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(build_fixtures, "AOI_LON", BORNHOLM_LON)
        monkey.setattr(build_fixtures, "AOI_LAT", BORNHOLM_LAT)

        with pytest.raises(land_guard.LandPlacementError) as error:
            build_fixtures.build_sc01(tmp_path)
    finally:
        monkey.undo()

    assert "release point" in str(error.value)
    assert not list(tmp_path.iterdir()), "a refused build left artefacts behind"
