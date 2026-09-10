"use client";

import { useState } from "react";

import { DemoController } from "@/components/demo/DemoController";
import { Scrubber } from "@/components/demo/Scrubber";
import { SeaMap } from "@/components/map/SeaMap";
import { AttributionPanel } from "@/components/panels/AttributionPanel";
import { DetectionPanel } from "@/components/panels/DetectionPanel";
import { ProvenanceBadges, SourceBadge } from "@/components/ui/Badge";
import type { ScenarioCode } from "@/lib/data";
import { clock } from "@/lib/format";
import { useConsole } from "@/lib/useConsole";

export default function Console() {
  const [scenario, setScenario] = useState<ScenarioCode>("SC-01");
  const [showDensity, setShowDensity] = useState(true);
  const [showTracks, setShowTracks] = useState(true);
  const state = useConsole(scenario);
  const { data, land, loading, error, frame, culprit, culpritIgnited } = state;

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-sea-abyss">
      {data ? (
        <SeaMap
          scene={data.scene}
          slick={data.detection?.geom.coordinates ?? null}
          drift={data.drift}
          frame={frame}
          ais={data.ais}
          land={land}
          culprit={culprit}
          culpritIgnited={culpritIgnited}
          headTime={state.headTime}
          showDensity={showDensity}
          showTracks={showTracks}
        />
      ) : (
        <div className="absolute inset-0 bg-sea-deep" />
      )}

      {/* header */}
      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start gap-3 p-3">
        <div className="panel pointer-events-auto flex items-center gap-3 px-3 py-2">
          <div>
            <h1 className="text-small font-semibold tracking-tight text-ink-bright">
              VARUNA
            </h1>
            <p className="text-micro text-ink-dim">Reverse-drift vessel attribution</p>
          </div>
          {data ? (
            <div className="border-l border-chart-line/60 pl-3">
              <div className="text-micro text-ink-dim">{data.scene.name}</div>
              <div className="tnum text-micro text-ink">{clock(data.scene.acq_time)}</div>
            </div>
          ) : null}
          {data ? <SourceBadge source={data.script.source} /> : null}
        </div>

        {data && data.scene.badges.length > 0 ? (
          <div className="panel pointer-events-auto px-2.5 py-2">
            <ProvenanceBadges badges={data.scene.badges} />
          </div>
        ) : null}

        <div className="panel pointer-events-auto ml-auto flex items-center gap-2 px-2.5 py-2">
          <Toggle label="Origin field" on={showDensity} onChange={setShowDensity} />
          <Toggle label="AIS tracks" on={showTracks} onChange={setShowTracks} />
        </div>
      </header>

      {/* left rail */}
      <div className="absolute bottom-24 left-3 top-[4.5rem] z-10 flex w-[21rem] flex-col gap-3">
        {data ? (
          <DetectionPanel
            scene={data.scene}
            detection={data.detection}
            className="max-h-full"
          />
        ) : null}
      </div>

      {/* right rail */}
      <div className="absolute bottom-24 right-3 top-[4.5rem] z-10 flex w-[22rem] flex-col gap-3">
        <DemoController
          console={state}
          scenario={scenario}
          onScenarioChange={setScenario}
          className="max-h-[45%]"
        />
        {data ? (
          <AttributionPanel
            attribution={data.attribution}
            vessels={data.ais?.vessels ?? []}
            culpritIgnited={culpritIgnited}
            className="min-h-0 flex-1"
          />
        ) : null}
      </div>

      {/* scrubber */}
      <div className="absolute inset-x-3 bottom-3 z-10">
        <Scrubber console={state} />
      </div>

      {loading ? (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-sea-abyss/70">
          <p className="text-tiny text-ink-dim">Reading the fixture set…</p>
        </div>
      ) : null}

      {error ? (
        <div className="absolute left-1/2 top-20 z-30 -translate-x-1/2">
          <div className="panel max-w-md px-3 py-2">
            <div className="text-tiny font-semibold text-hazard">
              The fixture set could not be read
            </div>
            <p className="mt-1 text-tiny leading-relaxed text-ink">{error}</p>
            <p className="mt-1 text-micro leading-relaxed text-ink-dim">
              The console reads data/fixtures through the dev server. Check that it is
              running from the repository root.
            </p>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function Toggle({
  label,
  on,
  onChange,
}: {
  label: string;
  on: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      aria-pressed={on}
      className={`rounded-chart border px-2 py-1 text-micro transition-colors ${
        on
          ? "border-survey/50 bg-survey/10 text-survey"
          : "border-chart-line/70 text-ink-faint hover:text-ink-dim"
      }`}
    >
      {label}
    </button>
  );
}
