"use client";

import { useMemo, useState } from "react";
import DeckGL from "deck.gl";
import Map from "react-map-gl/maplibre";
import type { StyleSpecification } from "maplibre-gl";

import "maplibre-gl/dist/maplibre-gl.css";

import { MAP_INITIAL } from "@/lib/config";
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

export function SeaMap(props: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

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
    <div className="absolute inset-0">
      <DeckGL
        initialViewState={MAP_INITIAL}
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
