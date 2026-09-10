/**
 * Geodesy for display.
 *
 * The drift engine integrates in a local UTM zone and stores geometry in
 * EPSG:4326 (§8), but `drift.frames[].density_cells` stay on the engine's
 * EPSG:32633 grid as [ix, iy, mass]. Drawing that field needs an inverse
 * transverse Mercator, so it lives here rather than pulling in proj4 — §9
 * closes the door on new dependencies, and this is forty lines of Snyder.
 */

import type { DensityGrid } from "@/types/domain";

const A = 6378137.0;
const F = 1 / 298.257223563;
const K0 = 0.9996;
const E2 = F * (2 - F);
const EP2 = E2 / (1 - E2);
const E1 = (1 - Math.sqrt(1 - E2)) / (1 + Math.sqrt(1 - E2));

const DEG = 180 / Math.PI;

/** Central meridian of a northern-hemisphere UTM zone, in degrees. */
function centralMeridian(zone: number): number {
  return (zone - 1) * 6 - 180 + 3;
}

/** EPSG:326xx easting/northing (metres) to [lon, lat] degrees. Snyder 8-x. */
export function utmToLonLat(
  easting: number,
  northing: number,
  zone: number,
): [number, number] {
  const x = easting - 500000;
  const m = northing / K0;
  const mu =
    m / (A * (1 - E2 / 4 - (3 * E2 * E2) / 64 - (5 * E2 * E2 * E2) / 256));

  const e1_2 = E1 * E1;
  const e1_3 = e1_2 * E1;
  const e1_4 = e1_3 * E1;

  const phi1 =
    mu +
    ((3 * E1) / 2 - (27 * e1_3) / 32) * Math.sin(2 * mu) +
    ((21 * e1_2) / 16 - (55 * e1_4) / 32) * Math.sin(4 * mu) +
    ((151 * e1_3) / 96) * Math.sin(6 * mu) +
    ((1097 * e1_4) / 512) * Math.sin(8 * mu);

  const sinPhi1 = Math.sin(phi1);
  const cosPhi1 = Math.cos(phi1);
  const tanPhi1 = Math.tan(phi1);

  const c1 = EP2 * cosPhi1 * cosPhi1;
  const t1 = tanPhi1 * tanPhi1;
  const denom = Math.sqrt(1 - E2 * sinPhi1 * sinPhi1);
  const n1 = A / denom;
  const r1 = (A * (1 - E2)) / (denom * denom * denom);
  const d = x / (n1 * K0);

  const d2 = d * d;
  const d3 = d2 * d;
  const d4 = d3 * d;
  const d5 = d4 * d;
  const d6 = d5 * d;

  const lat =
    phi1 -
    ((n1 * tanPhi1) / r1) *
      (d2 / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * EP2) * d4) / 24 +
        ((61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * EP2 - 3 * c1 * c1) * d6) /
          720);

  const lon =
    (d -
      ((1 + 2 * t1 + c1) * d3) / 6 +
      ((5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * EP2 + 24 * t1 * t1) * d5) / 120) /
    cosPhi1;

  return [centralMeridian(zone) + lon * DEG, lat * DEG];
}

/** "EPSG:32633" -> 33. Returns null for anything that is not northern UTM. */
export function utmZoneOf(crs: string): number | null {
  const match = /^EPSG:326(\d{2})$/.exec(crs);
  if (!match) {
    return null;
  }
  const zone = Number(match[1]);
  return zone >= 1 && zone <= 60 ? zone : null;
}

/** The four corners of grid cell (ix, iy) as a lon/lat ring. */
export function cellPolygon(
  grid: DensityGrid,
  ix: number,
  iy: number,
  zone: number,
): [number, number][] {
  const x0 = grid.x0_m + ix * grid.cell_size_m;
  const y0 = grid.y0_m + iy * grid.cell_size_m;
  const x1 = x0 + grid.cell_size_m;
  const y1 = y0 + grid.cell_size_m;
  return [
    utmToLonLat(x0, y0, zone),
    utmToLonLat(x1, y0, zone),
    utmToLonLat(x1, y1, zone),
    utmToLonLat(x0, y1, zone),
  ];
}

/** Agreement tolerance between the two centroids, in kilometres. */
const CENTROID_TOLERANCE_KM = 2;

/**
 * Confirms the reprojected density field actually lands on the cloud it was
 * built from.
 *
 * A silently misplaced probability field would put engine output in the wrong
 * part of the sea, which is exactly the class of thing §2.2 forbids. The test
 * is that the field's mass-weighted centroid coincides with the particle
 * centroid: they are two representations of the same cloud, so any error in
 * the projection separates them by tens of kilometres while a correct one
 * leaves them within metres.
 *
 * Comparing grid extent against particle extent would not work — the grid is
 * sized once for the whole run, so a frame whose cloud has drifted away from
 * the run's centre would fail a bounds check while being perfectly placed.
 */
export function gridAgreesWithParticles(
  grid: DensityGrid,
  zone: number,
  particles: [number, number][],
  cells: [number, number, number][],
): boolean {
  if (particles.length === 0 || cells.length === 0) {
    return false;
  }

  let mass = 0;
  let cellLon = 0;
  let cellLat = 0;
  for (const [ix, iy, value] of cells) {
    const [lon, lat] = utmToLonLat(
      grid.x0_m + (ix + 0.5) * grid.cell_size_m,
      grid.y0_m + (iy + 0.5) * grid.cell_size_m,
      zone,
    );
    mass += value;
    cellLon += lon * value;
    cellLat += lat * value;
  }
  if (mass <= 0) {
    return false;
  }
  cellLon /= mass;
  cellLat /= mass;

  let sumLon = 0;
  let sumLat = 0;
  for (const [lon, lat] of particles) {
    sumLon += lon;
    sumLat += lat;
  }
  const particleLon = sumLon / particles.length;
  const particleLat = sumLat / particles.length;

  // Degrees to kilometres at this latitude, good enough for a sanity bound.
  const kmPerDegLat = 111.32;
  const kmPerDegLon = kmPerDegLat * Math.cos((particleLat * Math.PI) / 180);
  const dx = (cellLon - particleLon) * kmPerDegLon;
  const dy = (cellLat - particleLat) * kmPerDegLat;

  return Math.hypot(dx, dy) <= CENTROID_TOLERANCE_KM;
}
