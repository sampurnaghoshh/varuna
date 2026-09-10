/** Console tunables. Nothing here changes engine output — display only. */

/**
 * Draw the bundled coastline.
 *
 * SC-01's authored slick and its particle cloud overlap Bornholm — the fixture
 * geometry was placed without coastline awareness, so 80 of 116 slick vertices
 * and ~90% of the particles at t* fall on the island. The fixture set is marked
 * provisional and not §9-frozen, so a scripts session can re-place SC-01 over
 * open water with `make fixtures`. Until then this flag is the escape hatch:
 * set it false to drop the land layer entirely rather than show a slick on an
 * island. SC-02 and SC-03 are both over open water and unaffected.
 */
export const SHOW_LAND = true;

/** Seconds of wall clock for a full 12 h rewind at 1x. */
export const REWIND_DURATION_S = 9;

/** deck.gl GPU tween between two real engine snapshots, in ms. */
export const PARTICLE_TRANSITION_MS = 420;

/** How long an AIS track trails behind the scrubber head, in minutes. */
export const TRACK_TRAIL_MIN = 240;

export const MAP_INITIAL = {
  longitude: 15.0,
  latitude: 55.25,
  zoom: 8.2,
  pitch: 0,
  bearing: 0,
} as const;

/**
 * The map canvas is the full window, but the panels float on top of it, so the
 * only part a judge can actually see is the gap between the rails. Fitting a
 * scene to the canvas centres it under the right-hand rail; these paddings
 * push the fit into the visible gap instead.
 *
 * They mirror app/page.tsx: left rail left-3 + w-[21rem], right rail right-3 +
 * w-[22rem], header top-[4.5rem], scrubber bottom-24. Keep them in step with
 * that layout — if a rail is resized, this is the other half of the change.
 */
export const MAP_FIT_PADDING_PX = {
  left: 12 + 21 * 16 + 16,
  right: 12 + 22 * 16 + 16,
  top: 4.5 * 16 + 16,
  bottom: 96 + 16,
} as const;

/** Never zoom past this when a scene's geometry is small. */
export const MAP_FIT_MAX_ZOOM = 11;
