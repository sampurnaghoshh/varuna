"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL, { WebMercatorViewport } from "deck.gl";
import Map from "react-map-gl/maplibre";
import type { StyleSpecification } from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

import {
  MAP_FIT_MAX_ZOOM,
  MAP_FIT_PADDING_PX,
  MAP_INITIAL,
  TRACK_TRAIL_MIN,
} from "@/lib/config";
import { buildLayers } from "@/components/map/layers";
import type { LayerInputs } from "@/components/map/layers";

/**
 * A style with no sources at all.
 *
 * §2.1 requires the demo to run with no internet, so the map must never reach
 * for a tile server. Everything above the water — coastline, footprint, drift,
 * AIS — is a deck.gl layer drawn from bundled data.
 */
const NO_NETWORK_STYLE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    {
      id: "sea",
      type: "background",
      paint: { "background-color": "#08151E" },
    },
  ],
};

type Props = Omit<LayerInputs, "onVesselHover"> & {
  onVesselHover?: (mmsi: number | null) => void;
};

type Bounds = { west: number; south: number; east: number; north: number };

function grow(bounds: Bounds | null, lon: number, lat: number): Bounds {
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return bounds as Bounds;
  if (!bounds) return { west: lon, south: lat, east: lon, north: lat };
  return {
    west: Math.min(bounds.west, lon),
    south: Math.min(bounds.south, lat),
    east: Math.max(bounds.east, lon),
    north: Math.max(bounds.north, lat),
  };
}

/**
 * Everything the scenario is about: the slick, the whole origin cone across all
 * snapshots, and the culprit's track where it is drawn. The scene footprint is
 * the fallback for a clean scene (SC-03), which has none of the three.
 */
function scenarioBounds(props: Props): Bounds | null {
  let bounds: Bounds | null = null;
  const addRings = (rings: number[][][]) => {
    for (const ring of rings) {
      for (const point of ring) {
        bounds = grow(bounds, point[0]!, point[1]!);
      }
    }
  };

  if (props.slick) addRings(props.slick);
  if (props.drift) {
    for (const frame of props.drift.frames) addRings(frame.hull.coordinates);
  }
  if (props.culprit && props.ais) {
    const vessel = props.ais.vessels.find((v) => v.mmsi === props.culprit!.mmsi);
    // Only the stretch of track the map ever draws at the moment the culprit
    // ignites — the trail behind t★. Fitting the vessel's whole AIS history
    // would frame a 270 km transit and shrink the slick to nothing.
    const tStar = props.culprit.channels.t_star;
    const end = tStar ? new Date(tStar).getTime() : null;
    const start = end === null ? null : end - TRACK_TRAIL_MIN * 60_000;
    for (const position of vessel?.positions ?? []) {
      const at = new Date(position.ts).getTime();
      if (end !== null && start !== null && (at < start || at > end)) continue;
      bounds = grow(bounds, position.lon, position.lat);
    }
  }
  if (!bounds) addRings(props.scene.footprint.coordinates);
  return bounds;
}

export function SeaMap(props: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry?.contentRect;
      if (box) setSize({ width: box.width, height: box.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  /**
   * Per-scenario framing. Without it the shared viewport centres SC-01's slick
   * and origin cone underneath the attribution rail, so the one thing the
   * scenario exists to show is the one thing behind a panel.
   */
  const viewState = useMemo(() => {
    const bounds = scenarioBounds(props);
    if (!bounds || !size) return MAP_INITIAL;

    const { left, right, top, bottom } = MAP_FIT_PADDING_PX;
    // fitBounds throws if the padding leaves no room; a window too narrow for
    // the rails falls back to the shared viewport rather than failing the map.
    if (left + right >= size.width - 40 || top + bottom >= size.height - 40) {
      return MAP_INITIAL;
    }

    // A degenerate box (a single point) has no extent to fit — pad it slightly.
    const pad = 0.005;
    const fitted = new WebMercatorViewport(size).fitBounds(
      [
        [Math.min(bounds.west, bounds.east - pad), Math.min(bounds.south, bounds.north - pad)],
        [Math.max(bounds.east, bounds.west + pad), Math.max(bounds.north, bounds.south + pad)],
      ],
      { padding: { left, right, top, bottom } },
    );

    return {
      longitude: fitted.longitude,
      latitude: fitted.latitude,
      zoom: Math.min(fitted.zoom, MAP_FIT_MAX_ZOOM),
      pitch: 0,
      bearing: 0,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.scene, props.slick, props.drift, props.culprit, props.ais, size]);

  const layers = useMemo(
    () =>
      buildLayers({
        ...props,
        onVesselHover: (mmsi) => {
          setHovered(mmsi);
          props.onVesselHover?.(mmsi);
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      props.scene,
      props.slick,
      props.drift,
      props.frame,
      props.ais,
      props.land,
      props.culprit,
      props.culpritIgnited,
      props.headTime,
      props.showDensity,
      props.showTracks,
    ],
  );

  const hoveredVessel = useMemo(
    () => props.ais?.vessels.find((v) => v.mmsi === hovered) ?? null,
    [props.ais, hovered],
  );

  return (
    <div className="absolute inset-0" ref={container}>
      <DeckGL
        initialViewState={viewState}
        controller={{ dragRotate: false, touchRotate: false }}
        layers={layers}
        getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}
      >
        <Map mapStyle={NO_NETWORK_STYLE} attributionControl={false} />
      </DeckGL>

      {hoveredVessel ? (
        <div className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-chart border border-chart-edge/70 bg-sea-deep/95 px-2.5 py-1.5 shadow-none backdrop-blur">
          <div className="flex items-center gap-2">
            <span className="text-tiny font-semibold text-ink-bright">
              {hoveredVessel.name}
            </span>
            <span className="tnum text-micro text-ink-dim">MMSI {hoveredVessel.mmsi}</span>
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-micro text-ink-dim">
            <span>{hoveredVessel.type}</span>
            <span className="tnum">{hoveredVessel.length_m} m</span>
            <span>{hoveredVessel.flag}</span>
          </div>
          {hoveredVessel.badge ? (
            <div className="mt-1 text-micro text-caution">{hoveredVessel.badge}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
