/**
 * Display formatting. Units live in the names (§8), and nothing here rounds a
 * value the reader is meant to judge without keeping enough precision to judge
 * it.
 */

/** "T-6h 30m" for a negative backward offset; "T+0" at the observation. */
export function offsetLabel(tOffsetMin: number): string {
  if (tOffsetMin === 0) {
    return "T+0";
  }
  const sign = tOffsetMin < 0 ? "−" : "+";
  const total = Math.abs(tOffsetMin);
  const hours = Math.floor(total / 60);
  const minutes = total % 60;
  return `T${sign}${hours}h ${String(minutes).padStart(2, "0")}m`;
}

/** 2024-03-13T21:10:00Z -> "13 Mar 21:10Z". Fixture times are all UTC. */
export function clock(iso: string): string {
  const at = new Date(iso);
  const day = String(at.getUTCDate()).padStart(2, "0");
  const month = at.toLocaleString("en-GB", { month: "short", timeZone: "UTC" });
  const hh = String(at.getUTCHours()).padStart(2, "0");
  const mm = String(at.getUTCMinutes()).padStart(2, "0");
  return `${day} ${month} ${hh}:${mm}Z`;
}

export function hhmm(iso: string): string {
  const at = new Date(iso);
  return `${String(at.getUTCHours()).padStart(2, "0")}:${String(
    at.getUTCMinutes(),
  ).padStart(2, "0")}Z`;
}

/** Likelihood ratios span orders of magnitude; keep the shape readable. */
export function lrLabel(lr: number): string {
  if (lr >= 100) return lr.toFixed(0);
  if (lr >= 10) return lr.toFixed(1);
  if (lr >= 0.01) return lr.toFixed(2);
  return lr.toExponential(1);
}

/** Signed, in nats, always with the sign so the direction is unmissable. */
export function nats(value: number): string {
  const sign = value < 0 ? "−" : "+";
  return `${sign}${Math.abs(value).toFixed(2)}`;
}

export function pct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number, digits = 2): string {
  return value.toFixed(digits);
}

/** Renders a nullable measurement, never a blank cell (§2.2). */
export function measurement(
  value: number | null | undefined,
  unit: string,
  digits = 2,
): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return unit ? `${value.toFixed(digits)} ${unit}` : value.toFixed(digits);
}

/** area_km2 -> "Area", mean_sigma0_db -> "Mean sigma0 (dB)". */
const FEATURE_LABELS: Record<string, string> = {
  area_km2: "Area",
  perimeter_km: "Perimeter",
  shape_complexity: "Shape complexity",
  major_axis_km: "Major axis",
  minor_axis_km: "Minor axis",
  eccentricity: "Eccentricity",
  orientation_deg: "Orientation",
  mean_sigma0_db: "Mean backscatter",
  std_sigma0_db: "Backscatter spread",
  contrast_db: "Contrast vs background",
  edge_gradient_mean: "Edge gradient",
  edge_gradient_std: "Edge gradient spread",
  glcm_homogeneity: "Texture homogeneity",
  glcm_contrast: "Texture contrast",
  glcm_entropy: "Texture entropy",
  wind_speed_ms: "Wind speed",
  distance_to_coast_km: "Distance to coast",
  n_ships_within_20km: "Ships within 20 km",
};

const FEATURE_UNITS: Record<string, [string, number]> = {
  area_km2: ["km²", 1],
  perimeter_km: ["km", 1],
  shape_complexity: ["", 2],
  major_axis_km: ["km", 2],
  minor_axis_km: ["km", 2],
  eccentricity: ["", 3],
  orientation_deg: ["°", 1],
  mean_sigma0_db: ["dB", 2],
  std_sigma0_db: ["dB", 2],
  contrast_db: ["dB", 2],
  edge_gradient_mean: ["", 3],
  edge_gradient_std: ["", 3],
  glcm_homogeneity: ["", 3],
  glcm_contrast: ["", 3],
  glcm_entropy: ["", 3],
  wind_speed_ms: ["m/s", 1],
  distance_to_coast_km: ["km", 1],
  n_ships_within_20km: ["", 0],
};

export function featureLabel(key: string): string {
  return FEATURE_LABELS[key] ?? key;
}

export function featureValue(key: string, value: number | null): string | null {
  if (value === null) {
    return null;
  }
  const spec = FEATURE_UNITS[key];
  if (!spec) {
    return String(value);
  }
  const [unit, digits] = spec;
  return unit ? `${value.toFixed(digits)} ${unit}` : value.toFixed(digits);
}
