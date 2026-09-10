import { GeoJsonLayer, PathLayer, PolygonLayer, ScatterplotLayer, TextLayer } from "deck.gl";
import type { Layer } from "deck.gl";

import { PARTICLE_TRANSITION_MS, SHOW_LAND, TRACK_TRAIL_MIN } from "@/lib/config";
import { cellPolygon, gridAgreesWithParticles, utmZoneOf } from "@/lib/geo";
import type {
  AisResponse,
  DriftFrame,
  DriftRun,
  LandCollection,
  Scene,
  Vessel,
} from "@/types/domain";
import type { Candidate } from "@/types/domain";

type RGBA = [number, number, number, number];

const SEA_EDGE: RGBA = [46, 84, 100, 190];
const LAND_FILL: RGBA = [26, 39, 33, 235];
const COAST: RGBA = [60, 98, 116, 235];
const SLICK: RGBA = [140, 123, 216, 70];
const SLICK_EDGE: RGBA = [163, 148, 232, 235];
const HULL_EDGE: RGBA = [121, 210, 232, 130];
const PARTICLE: RGBA = [121, 210, 232, 105];
const TRACK: RGBA = [64, 106, 124, 175];
const TRACK_LIVE: RGBA = [122, 155, 170, 225];
const HAZARD: RGBA = [232, 69, 155, 255];
const VESSEL: RGBA = [150, 179, 191, 230];

export interface LayerInputs {
  scene: Scene;
  slick: number[][][] | null;
  drift: DriftRun | null;
  frame: DriftFrame | null;
  ais: AisResponse | null;
  land: LandCollection | null;
  culprit: Candidate | null;
  culpritIgnited: boolean;
  headTime: string | null;
  showDensity: boolean;
  showTracks: boolean;
  onVesselHover: (mmsi: number | null) => void;
}

/** Position of a vessel at `at`, interpolated between its two nearest fixes. */
export function vesselAt(vessel: Vessel, at: number): [number, number] | null {
  const positions = vessel.positions;
  if (positions.length === 0) return null;

  const first = positions[0]!;
  const last = positions[positions.length - 1]!;
  if (at <= new Date(first.ts).getTime()) return [first.lon, first.lat];
  if (at >= new Date(last.ts).getTime()) return [last.lon, last.lat];

  for (let i = 0; i < positions.length - 1; i += 1) {
    const a = positions[i]!;
    const b = positions[i + 1]!;
    const ta = new Date(a.ts).getTime();
    const tb = new Date(b.ts).getTime();
    if (at >= ta && at <= tb) {
      const span = tb - ta;
      const f = span === 0 ? 0 : (at - ta) / span;
      return [a.lon + (b.lon - a.lon) * f, a.lat + (b.lat - a.lat) * f];
    }
  }
  return [last.lon, last.lat];
}

/** Track vertices within the trailing window ending at `at`. */
function trailOf(vessel: Vessel, at: number): [number, number][] {
  const from = at - TRACK_TRAIL_MIN * 60_000;
  const path = vessel.positions
    .filter((p) => {
      const t = new Date(p.ts).getTime();
      return t >= from && t <= at;
    })
    .map((p) => [p.lon, p.lat] as [number, number]);
  const head = vesselAt(vessel, at);
  if (head && path.length > 0) {
    path.push(head);
  }
  return path;
}

export function buildLayers(input: LayerInputs): Layer[] {
  const {
    scene,
    slick,
    drift,
    frame,
    ais,
    land,
    culprit,
    culpritIgnited,
    headTime,
    showDensity,
    showTracks,
    onVesselHover,
  } = input;

  const layers: Layer[] = [];
  const at = headTime ? new Date(headTime).getTime() : null;

  // --- basemap --------------------------------------------------------------

  if (SHOW_LAND && land) {
    layers.push(
      new GeoJsonLayer({
        id: "land",
        data: land as unknown as GeoJSON.FeatureCollection,
        filled: true,
        stroked: true,
        getFillColor: LAND_FILL,
        getLineColor: COAST,
        getLineWidth: 1.4,
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,
        pickable: false,
      }),
    );
  }

  layers.push(
    new PolygonLayer({
      id: "footprint",
      data: [scene.footprint.coordinates[0] ?? []],
      getPolygon: (d: number[][]) => d,
      filled: false,
      stroked: true,
      getLineColor: SEA_EDGE,
      getLineWidth: 1,
      lineWidthUnits: "pixels",
      getDashArray: [6, 4],
      pickable: false,
    }),
  );

  // --- the reconstructed origin --------------------------------------------

  if (showDensity && drift && frame) {
    const zone = utmZoneOf(drift.density_grid.crs);
    // A misplaced probability field would put engine output in the wrong part
    // of the sea. Draw it only once the projection is shown to agree with the
    // particles it was built from (§2.2).
    if (
      zone !== null &&
      gridAgreesWithParticles(
        drift.density_grid,
        zone,
        frame.particles,
        frame.density_cells,
      )
    ) {
      const peak = frame.density_cells.reduce((max, cell) => Math.max(max, cell[2]), 0);
      layers.push(
        new PolygonLayer({
          id: `density-${frame.t_offset_min}`,
          data: frame.density_cells,
          getPolygon: (cell: [number, number, number]) =>
            cellPolygon(drift.density_grid, cell[0], cell[1], zone),
          filled: true,
          stroked: false,
          getFillColor: (cell: [number, number, number]): RGBA => {
            const share = peak > 0 ? cell[2] / peak : 0;
            const eased = Math.sqrt(share);
            return [70, 150, 190, Math.round(12 + eased * 105)];
          },
          pickable: false,
        }),
      );
    }
  }

  if (frame) {
    layers.push(
      new PolygonLayer({
        id: "hull",
        data: [frame.hull.coordinates[0] ?? []],
        getPolygon: (d: number[][]) => d,
        filled: false,
        stroked: true,
        getLineColor: HULL_EDGE,
        getLineWidth: 1.6,
        lineWidthUnits: "pixels",
        pickable: false,
        updateTriggers: { getPolygon: frame.t_offset_min },
      }),
    );

    layers.push(
      new ScatterplotLayer({
        id: "particles",
        data: frame.particles,
        getPosition: (d: [number, number]) => d,
        getFillColor: PARTICLE,
        getRadius: 190,
        radiusMinPixels: 1,
        radiusMaxPixels: 3,
        pickable: false,
        // Tweens between two snapshots the engine actually integrated. Every
        // number on screen still comes from the snapped frame.
        transitions: { getPosition: PARTICLE_TRANSITION_MS },
      }),
    );
  }

  // --- AIS ------------------------------------------------------------------

  if (ais && at !== null) {
    const culpritMmsi = culprit?.mmsi ?? null;

    if (showTracks) {
      layers.push(
        new PathLayer({
          id: "tracks-full",
          data: ais.vessels,
          getPath: (v: Vessel) =>
            v.positions.map((p) => [p.lon, p.lat] as [number, number]),
          getColor: TRACK,
          getWidth: 1,
          widthUnits: "pixels",
          widthMinPixels: 1,
          pickable: false,
        }),
      );

      layers.push(
        new PathLayer({
          id: "tracks-trail",
          data: ais.vessels,
          getPath: (v: Vessel) => trailOf(v, at),
          getColor: (v: Vessel): RGBA =>
            v.mmsi === culpritMmsi && culpritIgnited ? HAZARD : TRACK_LIVE,
          getWidth: (v: Vessel) => (v.mmsi === culpritMmsi && culpritIgnited ? 2.6 : 1.4),
          widthUnits: "pixels",
          widthMinPixels: 1,
          pickable: false,
          updateTriggers: {
            getPath: at,
            getColor: [culpritMmsi, culpritIgnited],
            getWidth: [culpritMmsi, culpritIgnited],
          },
        }),
      );
    }

    const marks = ais.vessels
      .map((vessel) => ({ vessel, at: vesselAt(vessel, at) }))
      .filter((mark): mark is { vessel: Vessel; at: [number, number] } => mark.at !== null);

    layers.push(
      new ScatterplotLayer({
        id: "vessels",
        data: marks,
        getPosition: (m: { at: [number, number] }) => m.at,
        getFillColor: (m: { vessel: Vessel }): RGBA =>
          m.vessel.mmsi === culpritMmsi && culpritIgnited ? HAZARD : VESSEL,
        getRadius: (m: { vessel: Vessel }) =>
          m.vessel.mmsi === culpritMmsi && culpritIgnited ? 6 : 3.4,
        radiusUnits: "pixels",
        radiusMinPixels: 3,
        stroked: true,
        getLineColor: [8, 21, 30, 220],
        lineWidthMinPixels: 1,
        pickable: true,
        onHover: (info) => {
          const hovered = info.object as { vessel: Vessel } | undefined;
          onVesselHover(hovered?.vessel.mmsi ?? null);
        },
        updateTriggers: {
          getPosition: at,
          getFillColor: [culpritMmsi, culpritIgnited],
          getRadius: [culpritMmsi, culpritIgnited],
        },
      }),
    );

    if (culpritMmsi !== null && culpritIgnited) {
      const mark = marks.find((m) => m.vessel.mmsi === culpritMmsi);
      if (mark) {
        layers.push(
          new TextLayer({
            id: "culprit-label",
            data: [mark],
            getPosition: (m: { at: [number, number] }) => m.at,
            getText: (m: { vessel: Vessel }) => m.vessel.name,
            getSize: 12,
            getColor: HAZARD,
            getPixelOffset: [0, -16],
            fontFamily: "Segoe UI, system-ui, sans-serif",
            fontWeight: 600,
            outlineWidth: 3,
            outlineColor: [8, 21, 30, 255],
            fontSettings: { sdf: true },
            getTextAnchor: "middle",
            pickable: false,
            updateTriggers: { getPosition: at },
          }),
        );
      }
    }
  }

  // --- the observation ------------------------------------------------------

  if (slick) {
    layers.push(
      new PolygonLayer({
        id: "slick",
        data: [slick[0] ?? []],
        getPolygon: (d: number[][]) => d,
        filled: true,
        stroked: true,
        getFillColor: SLICK,
        getLineColor: SLICK_EDGE,
        getLineWidth: 1.6,
        lineWidthUnits: "pixels",
        pickable: false,
      }),
    );
  }

  return layers;
}
