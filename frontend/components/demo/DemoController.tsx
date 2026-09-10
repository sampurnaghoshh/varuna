"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { SCENARIOS, type ScenarioCode } from "@/lib/data";
import type { Console } from "@/lib/useConsole";
import type { ScriptedEvent, Stage } from "@/types/domain";

const STAGE_TONE: Record<Stage, string> = {
  SEGMENTING: "text-survey",
  DISCRIMINATING: "text-slick",
  REWINDING: "text-survey",
  FUSING: "text-caution",
  DONE: "text-clear",
  ERROR: "text-hazard",
};

const SCENARIO_NAMES: Record<ScenarioCode, string> = {
  "SC-01": "Baltic Night Discharge",
  "SC-02": "The Look-alike Trap",
  "SC-03": "Clean Sea",
};

/**
 * Replays a scenario on its authored pacing (§12) and drives the rewind with
 * it, so the narration and the particle cloud move together.
 *
 * The events are the same shape the §7 socket will push. When the backend is
 * wired, this component swaps its source from the script to the WebSocket and
 * keeps the rest.
 */
export function DemoController({
  console: state,
  scenario,
  onScenarioChange,
  className = "",
}: {
  console: Console;
  scenario: ScenarioCode;
  onScenarioChange: (code: ScenarioCode) => void;
  className?: string;
}) {
  const { data, seekToOffset, rewind, setPlaying } = state;
  const [running, setRunning] = useState(false);
  const [fired, setFired] = useState<ScriptedEvent[]>([]);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  const clearTimers = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  const stop = useCallback(() => {
    clearTimers();
    setRunning(false);
  }, [clearTimers]);

  const reset = useCallback(() => {
    stop();
    setFired([]);
    rewind();
  }, [stop, rewind]);

  // Clearing on unmount also covers React 18 StrictMode's double-invoke in dev.
  useEffect(() => clearTimers, [clearTimers]);

  useEffect(() => {
    clearTimers();
    setRunning(false);
    setFired([]);
  }, [scenario, clearTimers]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [fired.length]);

  const run = useCallback(() => {
    const script = data?.script;
    if (!script) return;

    clearTimers();
    setFired([]);
    rewind();
    setPlaying(false);
    setRunning(true);

    script.events.forEach((event) => {
      const timer = setTimeout(() => {
        setFired((previous) => [...previous, event]);

        // The rewind stages carry the snapshot the narration is describing.
        const offset = event.payload.t_offset_min;
        if (typeof offset === "number") {
          seekToOffset(offset);
        }
        if (event.stage === "DONE") {
          setRunning(false);
        }
      }, event.playback_offset_ms);
      timers.current.push(timer);
    });
  }, [data, clearTimers, rewind, seekToOffset, setPlaying]);

  const latest = fired[fired.length - 1] ?? null;
  const total = data?.script.total_playback_ms ?? 0;

  return (
    <div className={`panel flex min-h-0 flex-col ${className}`}>
      <div className="shrink-0 border-b border-chart-line/60 px-3 py-2">
        <div className="flex items-center gap-1">
          {SCENARIOS.map((code) => (
            <button
              key={code}
              type="button"
              onClick={() => onScenarioChange(code)}
              className={`tnum rounded-chart border px-2 py-1 text-micro transition-colors ${
                code === scenario
                  ? "border-survey/60 bg-survey/12 text-survey"
                  : "border-chart-line/70 text-ink-dim hover:border-chart-edge hover:text-ink"
              }`}
            >
              {code}
            </button>
          ))}
        </div>
        <div className="mt-1.5 text-tiny font-semibold text-ink-bright">
          {SCENARIO_NAMES[scenario]}
        </div>

        <div className="mt-2 flex items-center gap-2">
          <button
            type="button"
            onClick={running ? stop : run}
            disabled={!data?.script}
            className="rounded-chart border border-survey/50 bg-survey/10 px-2.5 py-1 text-micro text-survey transition-colors hover:bg-survey/20 disabled:opacity-40"
          >
            {running ? "Stop" : "Run scenario"}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-chart border border-chart-edge/70 px-2 py-1 text-micro text-ink-dim transition-colors hover:border-survey/50 hover:text-survey"
          >
            Reset
          </button>
          {total > 0 ? (
            <span className="tnum ml-auto text-micro text-ink-faint">
              {Math.round(total / 1000)}s
            </span>
          ) : null}
        </div>

        {latest ? (
          <div className="mt-2">
            <div className="h-1 overflow-hidden rounded-full bg-chart-line/60">
              <div
                className="h-full bg-survey transition-[width] duration-700 ease-out"
                style={{ width: `${latest.progress * 100}%` }}
              />
            </div>
          </div>
        ) : null}
      </div>

      <div ref={logRef} className="scrollable min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {fired.length === 0 ? (
          <p className="py-2 text-tiny leading-relaxed text-ink-dim">
            Run the scenario to watch the pipeline work through it stage by stage. The
            rewind follows the narration.
          </p>
        ) : (
          <ol className="space-y-2">
            {fired.map((event, index) => (
              <li key={`${event.stage}-${event.playback_offset_ms}`} className="flex gap-2">
                <span className="tnum shrink-0 pt-px text-micro text-ink-faint">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  <div className={`text-micro font-semibold ${STAGE_TONE[event.stage]}`}>
                    {event.stage}
                  </div>
                  <p className="text-tiny leading-relaxed text-ink">{event.message}</p>
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
