"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  attribute,
  demoRun,
  detect,
  driftBackward,
  getScene,
  landmass,
  vesselTracks,
  type ScenarioCode,
} from "@/lib/data";
import { REWIND_DURATION_S } from "@/lib/config";
import type {
  AisResponse,
  AttributionResponse,
  DemoScript,
  Detection,
  DriftFrame,
  DriftRun,
  LandCollection,
  Scene,
} from "@/types/domain";

export interface ScenarioData {
  scene: Scene;
  detection: Detection | null;
  drift: DriftRun | null;
  attribution: AttributionResponse | null;
  ais: AisResponse | null;
  script: DemoScript;
}

/**
 * Loads one scenario's whole fixture set and owns the rewind clock.
 *
 * The scrubber is snapped to real snapshot indices rather than free time: every
 * frame the console draws is an engine state that was actually integrated, and
 * every number beside it comes from that same frame (§2.2). Smoothness is
 * deck.gl's problem, not the data's.
 */
export function useConsole(scenario: ScenarioCode) {
  const [data, setData] = useState<ScenarioData | null>(null);
  const [land, setLand] = useState<LandCollection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    setFrameIndex(0);
    setPlaying(false);

    (async () => {
      try {
        const [sceneResponse, detectResponse, ais, script] = await Promise.all([
          getScene(scenario),
          detect(scenario),
          vesselTracks(scenario),
          demoRun(scenario),
        ]);
        const detection = detectResponse.detections[0] ?? null;
        const [drift, attribution] = await Promise.all([
          detection ? driftBackward(detection.id) : Promise.resolve(null),
          detection ? attribute(detection.id) : Promise.resolve(null),
        ]);
        if (!live) return;
        setData({
          scene: sceneResponse.scene,
          detection,
          drift,
          attribution,
          ais,
          script,
        });
      } catch (cause) {
        if (!live) return;
        setError(cause instanceof Error ? cause.message : "Could not read the fixture set.");
      } finally {
        if (live) setLoading(false);
      }
    })();

    return () => {
      live = false;
    };
  }, [scenario]);

  useEffect(() => {
    let live = true;
    landmass()
      .then((collection) => {
        if (live) setLand(collection);
      })
      .catch(() => {
        // Cartographic context only; the console is complete without it.
      });
    return () => {
      live = false;
    };
  }, []);

  const frames: DriftFrame[] = useMemo(() => data?.drift?.frames ?? [], [data]);

  const frame = frames[Math.min(frameIndex, Math.max(0, frames.length - 1))] ?? null;

  /** Model time at the scrubber head, or the acquisition when there is no run. */
  const headTime = frame?.t ?? data?.scene.acq_time ?? null;

  const tOffsetMin = frame?.t_offset_min ?? 0;

  // --- the rewind clock -----------------------------------------------------

  const rafRef = useRef<number | null>(null);
  const lastStepRef = useRef(0);

  useEffect(() => {
    if (!playing || frames.length < 2) {
      return;
    }
    const perFrameMs = (REWIND_DURATION_S * 1000) / (frames.length - 1);
    lastStepRef.current = performance.now();

    const tick = (now: number) => {
      if (now - lastStepRef.current >= perFrameMs) {
        lastStepRef.current = now;
        setFrameIndex((index) => {
          if (index >= frames.length - 1) {
            setPlaying(false);
            return index;
          }
          return index + 1;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [playing, frames.length]);

  const seekToOffset = useCallback(
    (targetOffsetMin: number) => {
      if (frames.length === 0) return;
      let best = 0;
      let bestGap = Infinity;
      frames.forEach((candidate, index) => {
        const gap = Math.abs(candidate.t_offset_min - targetOffsetMin);
        if (gap < bestGap) {
          bestGap = gap;
          best = index;
        }
      });
      setFrameIndex(best);
    },
    [frames],
  );

  const rewind = useCallback(() => {
    setFrameIndex(0);
    setPlaying(false);
  }, []);

  // --- t*, the moment the culprit ignites -----------------------------------

  const culprit = useMemo(() => {
    const attribution = data?.attribution;
    if (!attribution || !attribution.issued) {
      return null;
    }
    return attribution.candidates.find((c) => c.mmsi === attribution.culprit_mmsi) ?? null;
  }, [data]);

  const tStarOffsetMin = useMemo(() => {
    if (!culprit?.channels.t_star || !data?.drift) {
      return null;
    }
    const tStar = new Date(culprit.channels.t_star).getTime();
    const t0 = new Date(data.drift.t0).getTime();
    return Math.round((tStar - t0) / 60000);
  }, [culprit, data]);

  /** True once the scrubber has rewound to or past the release moment. */
  const culpritIgnited =
    tStarOffsetMin !== null && frame !== null && frame.t_offset_min <= tStarOffsetMin;

  return {
    data,
    land,
    loading,
    error,
    frames,
    frame,
    frameIndex,
    setFrameIndex,
    tOffsetMin,
    headTime,
    playing,
    setPlaying,
    rewind,
    seekToOffset,
    culprit,
    tStarOffsetMin,
    culpritIgnited,
  };
}

export type Console = ReturnType<typeof useConsole>;
