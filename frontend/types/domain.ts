/**
 * Mirrors the §7 response shapes and the fixture set they were captured from.
 * Keep in sync with backend/app/schemas/ (§8).
 *
 * Nullable fields are deliberate: the fixture set records what it could not
 * measure, and the console renders the reason rather than a blank (§2.2).
 * Anything nullable here has a matching entry in `unavailable`.
 */

import type { PipelineEvent, Verdict } from "./pipeline";

export type { PipelineEvent, Stage, Verdict } from "./pipeline";

export type Source = "live" | "fixture";

export interface Envelope {
  source: Source;
  elapsed_ms: number;
}

/** Free-text reasons keyed by the field that is missing. */
export type Unavailable = Record<string, string>;

export interface Position {
  lon: number;
  lat: number;
}

export interface PolygonGeom {
  type: "Polygon";
  coordinates: number[][][];
}

// --- scenes -----------------------------------------------------------------

export interface Scene {
  id: string;
  product_id: string;
  name: string;
  sensor: string;
  acq_time: string;
  footprint: PolygonGeom;
  centre: Position;
  incidence_angle_deg: number | null;
  wind_speed_ms: number;
  wind_dir_deg: number;
  pixel_spacing_m: number | null;
  source: string;
  raster_path: string | null;
  badges: string[];
  unavailable: Unavailable;
}

export interface SceneResponse extends Envelope {
  scene: Scene;
}

// --- detection (§5.2) -------------------------------------------------------

export interface DetectionFeatures {
  area_km2: number | null;
  perimeter_km: number | null;
  shape_complexity: number | null;
  major_axis_km: number | null;
  minor_axis_km: number | null;
  eccentricity: number | null;
  orientation_deg: number | null;
  mean_sigma0_db: number | null;
  std_sigma0_db: number | null;
  contrast_db: number | null;
  edge_gradient_mean: number | null;
  edge_gradient_std: number | null;
  glcm_homogeneity: number | null;
  glcm_contrast: number | null;
  glcm_entropy: number | null;
  wind_speed_ms: number | null;
  distance_to_coast_km: number | null;
  n_ships_within_20km: number | null;
}

export interface Factor {
  feature: string;
  value: number;
  contribution: number;
  direction: "oil" | "look-alike";
  rationale: string;
  /** "shap" once a model is trained; "rule_based" in this build. Always shown. */
  basis: string;
}

export interface Detection {
  id: string;
  scene_id: string;
  geom: PolygonGeom;
  features: DetectionFeatures;
  feature_provenance: Record<string, string>;
  p_oil: number;
  class: string;
  /** §2.2 - never render p_oil without method and note beside it. */
  method: string;
  note: string;
  shap_factors: Factor[];
  factor_basis: string;
  base_p_oil: number;
  evidence_fraction: number;
  wind_gate_violated: boolean;
  wind_gate_reason: string;
  wind_gate_range_ms: number[];
  model_version: string | null;
  detector: string;
  unavailable: Unavailable;
  provenance: string;
  /** Present when the gate withheld attribution outright (SC-02). */
  attribution?: NotIssued;
}

export interface DetectResponse extends Envelope {
  scene_id: string;
  detections: Detection[];
}

// --- drift (§5.1) -----------------------------------------------------------

export interface DensityGrid {
  crs: string;
  x0_m: number;
  y0_m: number;
  cell_size_m: number;
  engine_cell_size_m: number;
  nx: number;
  ny: number;
  coarsen_factor: number;
  note: string;
  mass_kept_fraction: number;
  mass_retained_run_total: number;
}

/** [ix, iy, normalised mass] on the EPSG:32633 grid described by DensityGrid. */
export type DensityCell = [number, number, number];

export interface DriftFrame {
  t_offset_min: number;
  t: string;
  stretch_factor: number;
  stretch_factor_raw: number;
  n_particles: number;
  /** [lon, lat] per particle; the index is stable across frames. */
  particles: [number, number][];
  hull: PolygonGeom;
  hull_area_km2: number;
  density_cells: DensityCell[];
}

export interface DriftRun extends Envelope {
  run_id: string;
  mode: "backward" | "forward";
  t0: string;
  horizon_h: number;
  n_particles: number;
  windage: number;
  k_h_m2s: number;
  theta_dev_deg: number;
  dt_s: number;
  snapshot_interval_min: number;
  field_source: string;
  seed: number;
  n_frames: number;
  density_grid: DensityGrid;
  frames: DriftFrame[];
}

export interface DriftParams {
  horizon_h?: number;
  /** P1-11 precomputed variants exist in the fixture set; unused this build. */
  windage?: number;
  k_h?: number;
  theta_dev?: number;
}

// --- attribution (§5.3, §5.4) -----------------------------------------------

export interface Channels {
  mmsi: number;
  s1: number | null;
  s2: number | null;
  s3: number | null;
  s4: number | null;
  mass: number | null;
  t_star: string | null;
  stretch_factor: number | null;
  unavailable: Unavailable;
}

/** §5.4 log LR terms, in nats. These sum to log_lr. */
export interface TermsNats {
  s1: number;
  s2: number;
  s3: number;
  s4: number;
  prior: number;
}

export interface Candidate {
  rank: number;
  mmsi: number;
  name: string;
  log_lr: number;
  lr: number;
  posterior: number;
  verdict: Verdict;
  prior: number;
  terms_nats: TermsNats;
  channels: Channels;
}

/** A vessel that could not be scored. First-class, never hidden (§2.4). */
export interface Unrankable {
  mmsi: number;
  name: string;
  reason: string;
  verdict: Verdict;
  channels: Channels;
}

export interface Background {
  s1_bg: number;
  s2_bg: number;
  s3_bg: number;
}

export interface AttributionIssued extends Envelope {
  issued: true;
  verdict: Verdict;
  culprit_mmsi: number | null;
  windage: number;
  n_vessels_in_frame: number;
  n_ranked: number;
  frame_size_note: string;
  background: Background;
  none_of_the_above_posterior: number;
  decision_bands: Record<string, string>;
  candidates: Candidate[];
  unrankable: Unrankable[];
}

/** Not an error path. The system declining to accuse is a result (§2.4). */
export interface NotIssued {
  issued: false;
  verdict: Verdict;
  reason: string;
  recommendation: string;
  n_vessels_in_frame: number;
}

export type NotIssuedResponse = NotIssued & Envelope;
export type AttributionResponse = AttributionIssued | NotIssuedResponse;

// --- AIS --------------------------------------------------------------------

export interface AisPosition {
  ts: string;
  lon: number;
  lat: number;
  sog_kn: number;
  cog_deg: number;
  is_injected: boolean;
}

export interface Vessel {
  mmsi: number;
  name: string;
  type: string;
  canonical_type: string;
  length_m: number;
  width_m: number;
  flag: string;
  prior_detections: number;
  is_injected: boolean;
  /** §2.3 - rendered verbatim wherever the vessel appears. */
  badge: string | null;
  t_start: string;
  t_end: string;
  n_positions: number;
  gap_count: number;
  max_gap_min: number;
  positions: AisPosition[];
}

export interface AisResponse extends Envelope {
  provenance: string;
  note: string;
  badge: string;
  cadence_min: number;
  cadence_note: string;
  n_vessels: number;
  vessels: Vessel[];
}

// --- demo (§7 WS) -----------------------------------------------------------

export interface ScriptedEvent extends PipelineEvent {
  /** Authored demo pacing (§12), not a measured timing. */
  playback_offset_ms: number;
}

export interface DemoScript extends Envelope {
  scenario: string;
  timings_ms: Record<string, number>;
  timings_note: string;
  total_playback_ms: number;
  events: ScriptedEvent[];
}

// --- manifest ---------------------------------------------------------------

export interface ScenarioSummary {
  code: string;
  name: string;
  verdict?: string;
  culprit_mmsi?: number;
  lr?: number;
  n_vessels_in_frame?: number;
  n_ranked?: number;
  n_detections?: number;
  attribution_issued?: boolean;
  wind_gate_violated?: boolean;
  artefacts: string[];
}

export interface Manifest extends Envelope {
  provisional: boolean;
  frozen: boolean;
  why_provisional: string;
  data_tier: string;
  badges: string[];
  pending: string[];
  scenarios: ScenarioSummary[];
}

// --- basemap ----------------------------------------------------------------

export interface LandFeature {
  type: "Feature";
  properties: { render: "fill" | "stroke" };
  geometry:
    | { type: "Polygon"; coordinates: number[][][] }
    | { type: "LineString"; coordinates: number[][] };
}

export interface LandCollection {
  type: "FeatureCollection";
  properties: Record<string, unknown>;
  features: LandFeature[];
}
