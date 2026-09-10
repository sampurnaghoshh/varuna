"use client";

import { clock, num, offsetLabel } from "@/lib/format";
import type { Console } from "@/lib/useConsole";

/**
 * The rewind control.
 *
 * The slider indexes snapshots, not time, because every position it can stop
 * at is a state the solver actually integrated (§5.1 emits one every 30 min of
 * model time). The readouts beside it come from that same snapshot, so nothing
 * on screen is interpolated.
 */
export function Scrubber({ console: state }: { console: Console }) {
  const { frames, frame, frameIndex, setFrameIndex, playing, setPlaying, rewind } = state;
  const { tStarOffsetMin, culpritIgnited, culprit, data } = state;

  if (frames.length === 0 || !frame) {
    return (
      <div className="panel flex items-center gap-3 px-3 py-2.5">
        <p className="text-tiny text-ink-dim">
          {data?.detection
            ? "No drift run for this scene — the wind gate withheld it before the solver ran."
            : "Nothing to rewind. The scene is clean."}
        </p>
      </div>
    );
  }

  const last = frames.length - 1;
  const horizonMin = Math.abs(frames[last]?.t_offset_min ?? 0);
  const tStarFraction =
    tStarOffsetMin !== null && horizonMin > 0
      ? Math.abs(tStarOffsetMin) / horizonMin
      : null;

  return (
    <div className="panel px-3 py-2.5">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setPlaying(!playing)}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-chart border border-survey/50 bg-survey/10 text-survey transition-colors hover:bg-survey/20"
          aria-label={playing ? "Pause the rewind" : "Play the rewind"}
        >
          {playing ? (
            <svg viewBox="0 0 12 12" className="h-3 w-3" aria-hidden="true">
              <rect x="2" y="1.5" width="3" height="9" fill="currentColor" />
              <rect x="7" y="1.5" width="3" height="9" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 12 12" className="ml-0.5 h-3 w-3" aria-hidden="true">
              <path d="M2.5 1.5l8 4.5-8 4.5z" fill="currentColor" />
            </svg>
          )}
        </button>

        <button
          type="button"
          onClick={rewind}
          className="shrink-0 rounded-chart border border-chart-edge/70 px-2 py-1 text-micro text-ink-dim transition-colors hover:border-survey/50 hover:text-survey"
        >
          Back to observation
        </button>

        <div className="min-w-0 flex-1">
          <div className="relative">
            {/* t* tick — where the culprit's evidence peaks (§5.3 E1). */}
            {tStarFraction !== null ? (
              <div
                className="pointer-events-none absolute -top-[9px] z-10 -translate-x-1/2"
                style={{ left: `${tStarFraction * 100}%` }}
              >
                <div
                  className={`h-0 w-0 border-x-[4px] border-t-[5px] border-x-transparent ${
                    culpritIgnited ? "border-t-hazard" : "border-t-ink-faint"
                  }`}
                />
              </div>
            ) : null}

            <input
              type="range"
              min={0}
              max={last}
              step={1}
              value={frameIndex}
              onChange={(event) => {
                setPlaying(false);
                setFrameIndex(Number(event.target.value));
              }}
              aria-label="Rewind to a drift snapshot"
              className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-chart-line accent-survey
                [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:border [&::-webkit-slider-thumb]:border-sea-abyss
                [&::-webkit-slider-thumb]:bg-survey
                [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5
                [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0
                [&::-moz-range-thumb]:bg-survey"
            />
          </div>

          <div className="mt-1 flex items-center justify-between text-micro text-ink-faint">
            <span>observation</span>
            {tStarFraction !== null ? (
              <span className={culpritIgnited ? "text-hazard" : ""}>
                t★ {offsetLabel(tStarOffsetMin ?? 0)}
              </span>
            ) : null}
            <span>−{Math.round(horizonMin / 60)} h</span>
          </div>
        </div>

        <div className="shrink-0 text-right">
          <div className="tnum text-readout font-semibold text-ink-bright">
            {offsetLabel(frame.t_offset_min)}
          </div>
          <div className="tnum text-micro text-ink-dim">{clock(frame.t)}</div>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-chart-line/50 pt-2 text-micro text-ink-dim">
        <span>
          Snapshot{" "}
          <span className="tnum text-ink">
            {frameIndex + 1}/{frames.length}
          </span>
        </span>
        <span>
          Origin cone{" "}
          <span className="tnum text-ink">{num(frame.hull_area_km2, 1)} km²</span>
        </span>
        <span>
          Particles <span className="tnum text-ink">{frame.n_particles}</span>
        </span>
        <span title="§5.3 — observed major axis divided by the released patch length.">
          Stretch <span className="tnum text-ink">{num(frame.stretch_factor, 3)}</span>
        </span>
        {culprit && culpritIgnited ? (
          <span className="ml-auto text-hazard">
            {culprit.name} was here at t★
          </span>
        ) : (
          <span className="ml-auto text-ink-faint">
            Rewinding widens the cone — diffusion is symmetric in time (§5.1)
          </span>
        )}
      </div>
    </div>
  );
}
