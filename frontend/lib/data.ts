/**
 * The console's only data source.
 *
 * Every function here mirrors a §7 endpoint one-for-one. Components import
 * from this module and never touch fixture JSON directly, so wiring the live
 * backend is a change to the bodies below and nothing else — see the LIVE
 * comment on each function for the endpoint it becomes.
 *
 * §15 makes `source: "fixture"` the normal case during this build, not an
 * error state. Missing artefacts are typed results, never throws: SC-02 has no
 * drift run because the wind gate withheld one, and SC-03 has no detection at
 * all. Both are answers (§2.4, §8).
 */

import type {
  AisResponse,
  AttributionIssued,
  AttributionResponse,
  DemoScript,
  DetectResponse,
  DriftParams,
  DriftRun,
  Envelope,
  LandCollection,
  Manifest,
  Scene,
  SceneResponse,
} from "@/types/domain";

const FIXTURE_BASE = "/localdata/fixtures";
const GEO_BASE = "/localdata/geo";

/** Parsed fixtures, kept for the session. drift.json is 2.6 MB — parse once. */
const cache = new Map<string, Promise<unknown>>();

async function loadJson<T>(url: string): Promise<T> {
  const existing = cache.get(url);
  if (existing) {
    return existing as Promise<T>;
  }
  const pending = fetch(url, { cache: "no-store" }).then(async (response) => {
    if (!response.ok) {
      throw new FixtureMissing(url, response.status);
    }
    return (await response.json()) as unknown;
  });
  cache.set(url, pending);
  pending.catch(() => cache.delete(url));
  return pending as Promise<T>;
}

export class FixtureMissing extends Error {
  constructor(
    readonly url: string,
    readonly status: number,
  ) {
    super(`No fixture at ${url} (${status})`);
    this.name = "FixtureMissing";
  }
}

/** Reads one artefact, or null when the scenario does not have that stage. */
async function artefact<T>(scenario: string, name: string): Promise<T | null> {
  try {
    return await loadJson<T>(`${FIXTURE_BASE}/${scenario}/${name}.json`);
  } catch (error) {
    if (error instanceof FixtureMissing && error.status === 404) {
      return null;
    }
    throw error;
  }
}

function envelope<T extends object>(body: T, startedAt: number): T & Envelope {
  return {
    ...body,
    source: "fixture" as const,
    elapsed_ms: Math.max(1, Math.round(performance.now() - startedAt)),
  };
}

/** "SC-01-D1" and "SC-01" both belong to scenario SC-01. */
export function scenarioOf(id: string): string {
  const match = /^(SC-\d+)/.exec(id);
  return match?.[1] ?? id;
}

export const SCENARIOS = ["SC-01", "SC-02", "SC-03"] as const;
export type ScenarioCode = (typeof SCENARIOS)[number];

// --- §7 surface -------------------------------------------------------------

/** LIVE: GET /scenes */
export async function listScenes(): Promise<Manifest> {
  const started = performance.now();
  const manifest = await loadJson<Omit<Manifest, keyof Envelope>>(
    `${FIXTURE_BASE}/manifest.json`,
  );
  return envelope(manifest, started);
}

/** LIVE: GET /scenes/{id} */
export async function getScene(sceneId: string): Promise<SceneResponse> {
  const started = performance.now();
  const scene = await artefact<Scene>(scenarioOf(sceneId), "scene");
  if (!scene) {
    throw new FixtureMissing(sceneId, 404);
  }
  return envelope({ scene }, started);
}

/**
 * LIVE: POST /detect/{scene_id}
 *
 * SC-03 has no detection artefact because the scene is clean. Zero detections
 * is the result, not a failure.
 */
export async function detect(sceneId: string): Promise<DetectResponse> {
  const started = performance.now();
  const scenario = scenarioOf(sceneId);
  const detection = await artefact<DetectResponse["detections"][number]>(
    scenario,
    "detection",
  );
  return envelope(
    { scene_id: scenario, detections: detection ? [detection] : [] },
    started,
  );
}

/**
 * LIVE: POST /drift/backward
 *
 * `params` is accepted so the P1-11 windage variants can be wired later
 * without a refactor; this build always reads the default run.
 */
export async function driftBackward(
  detectionId: string,
  params: DriftParams = {},
): Promise<DriftRun | null> {
  const started = performance.now();
  const scenario = scenarioOf(detectionId);
  const name =
    params.windage === undefined
      ? "drift"
      : `drift.windage-${params.windage.toFixed(3)}`;
  const run = await artefact<Omit<DriftRun, keyof Envelope | "run_id">>(
    scenario,
    name,
  );
  if (!run) {
    return null;
  }
  return envelope({ ...run, run_id: `${scenario}-BACK` }, started);
}

/**
 * LIVE: GET /drift/{run_id}/frames?step_min=30
 *
 * The fixture set is already at the §5.1 30-minute snapshot interval, so this
 * filters rather than resamples.
 */
export async function driftFrames(
  runId: string,
  stepMin = 30,
): Promise<DriftRun | null> {
  const run = await driftBackward(runId);
  if (!run) {
    return null;
  }
  if (stepMin === run.snapshot_interval_min) {
    return run;
  }
  const frames = run.frames.filter((frame) => frame.t_offset_min % stepMin === 0);
  return { ...run, frames, n_frames: frames.length };
}

/**
 * LIVE: POST /attribute/{detection_id}
 *
 * Returns the not-issued block when the pipeline declined to name a vessel.
 * SC-02 carries that block inside its detection artefact, because the wind
 * gate stopped the run before an attribution file was ever written.
 */
export async function attribute(
  detectionId: string,
): Promise<AttributionResponse | null> {
  const started = performance.now();
  const scenario = scenarioOf(detectionId);

  const issued = await artefact<Omit<AttributionIssued, keyof Envelope | "issued">>(
    scenario,
    "attribution",
  );
  if (issued) {
    return envelope({ ...issued, issued: true as const }, started);
  }

  const detection = await artefact<DetectResponse["detections"][number]>(
    scenario,
    "detection",
  );
  const withheld = detection?.attribution;
  if (withheld) {
    return envelope({ ...withheld, issued: false as const }, started);
  }
  return null;
}

/** LIVE: GET /vessels/{mmsi}/track — the fixture set carries the whole frame. */
export async function vesselTracks(sceneId: string): Promise<AisResponse | null> {
  const started = performance.now();
  const ais = await artefact<Omit<AisResponse, keyof Envelope>>(
    scenarioOf(sceneId),
    "ais",
  );
  return ais ? envelope(ais, started) : null;
}

/**
 * LIVE: POST /demo/run/{code}, then the WS /ws/pipeline/{job_id} stream.
 *
 * The fixture carries the same event shape the socket will push, plus the
 * authored playback pacing the controller replays it on.
 */
export async function demoRun(code: string): Promise<DemoScript> {
  const started = performance.now();
  const script = await loadJson<Omit<DemoScript, keyof Envelope>>(
    `${FIXTURE_BASE}/${code}/ws.json`,
  );
  return envelope(script, started);
}

/** Basemap land. Cartographic context only — no pipeline stage reads it. */
export async function landmass(): Promise<LandCollection> {
  return loadJson<LandCollection>(`${GEO_BASE}/baltic_land.geojson`);
}
