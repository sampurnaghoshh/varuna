"""Application settings. Every tunable in §5 of CLAUDE.md lives here (§8).

Deliberately absent: the §5.4 decision bands (ln 10 / ln 100). §9 lists them as a
safety property rather than a tuning parameter, so they are module-level Final
constants in `app.attribution.fusion` and are not overridable by environment.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VARUNA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----------------------------------------------------------- runtime ----

    app_name: str = "VARUNA"
    env: Literal["dev", "demo"] = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]

    database_url: str = "postgresql+asyncpg://varuna:varuna@db:5432/varuna"
    redis_url: str = "redis://redis:6379/0"

    # ------------------------------------------------------------- paths ----

    data_dir: Path = Path("/data")
    scenes_dir: Path = Path("/data/scenes")
    ais_dir: Path = Path("/data/ais")
    models_dir: Path = Path("/data/models")
    fixtures_dir: Path = Path("/data/fixtures")

    # -------------------------------------------- §5.1 reverse-drift solver ----

    windage_alpha: float = Field(default=0.033, ge=0.020, le=0.040)
    theta_dev_deg: float = Field(default=0.0, ge=-20.0, le=20.0)
    drift_dt_s: int = Field(default=300, gt=0)
    drift_horizon_h: int = Field(default=12, gt=0)
    drift_n_particles: int = Field(default=5000, gt=0)
    k_h_m2s: float = Field(default=10.0, gt=0.0)
    drift_snapshot_interval_min: int = Field(default=30, gt=0)
    drift_seed: int = 42

    # Stokes drift is not shipped by every field source. Where it is absent it is
    # approximated as a fixed fraction of U10 aligned with the wind - the standard
    # first-order surface-Stokes relation.
    stokes_wind_fraction: float = Field(default=0.015, ge=0.0, le=0.05)

    # Synthetic field (tier 3): 2-3 mesoscale eddies, per §5.1.
    synthetic_n_eddies: int = Field(default=3, ge=2, le=3)
    synthetic_eddy_speed_ms: float = Field(default=0.35, gt=0.0)
    synthetic_wind_speed_ms: float = Field(default=7.0, gt=0.0)

    # --------------------------------------- §5.1 density / hull ----

    drift_cell_size_m: float = Field(default=200.0, gt=0.0)
    drift_kde_bandwidth_m: float = Field(default=500.0, gt=0.0)
    drift_grid_pad_m: float = Field(default=2000.0, ge=0.0)

    # Alpha-shape circumradius cutoff, in multiples of mean inter-particle
    # spacing. Measured on a 12 h backward cloud: 6 leaves ~11% of particles
    # outside the uncertainty hull, 9 keeps ~96% inside while still cutting ~20%
    # off the convex hull, so the concavities the hull exists to show survive.
    drift_hull_spacing_multiple: float = Field(default=9.0, gt=0.0)

    # The full 5000-particle 12 h run plus density must stay inside this budget:
    # §7 streams stage events live, and a run slower than the stage sequence makes
    # the demo unwatchable.
    drift_budget_s: float = Field(default=10.0, gt=0.0)

    # ------------------------------------------- §5.2 detection wind gate ----
    # Below 3 m/s the sea surface itself mimics oil; above 12 m/s slicks
    # disperse below detectability.

    wind_gate_min_ms: float = 3.0
    wind_gate_max_ms: float = 12.0

    # ------------------------------------- §5.2 CFAR dark-formation detector ----
    # `detection/classical.py` is the always-works fallback (§9), so every knob it
    # reads lives here rather than inline: a laptop that cannot load the U-Net
    # still has to produce polygons, and tuning it must not require an edit.

    # Oil damps capillary waves by roughly 3-10 dB against the surrounding sea.
    # 2.0 dB sits below the weakest of that range: the discriminator, not the
    # threshold, is what rejects the false alarms this admits.
    cfar_offset_db: float = Field(default=2.0, gt=0.0)
    # The background window must be wide enough that a slick occupies a small
    # fraction of it - otherwise the slick raises its own local mean and hides.
    cfar_window_px: int = Field(default=201, gt=2)
    # A window whose local contrast is below the sensor noise floor carries no
    # information, and thresholding it turns speckle into detections.
    cfar_min_local_std_db: float = Field(default=0.3, ge=0.0)
    cfar_open_px: int = Field(default=3, ge=0)
    cfar_close_px: int = Field(default=5, ge=0)

    # Below this a dark formation is speckle or a single-look artefact, not a
    # slick worth drifting: the §5.1 solver seeds 5000 particles in the polygon.
    detection_min_area_km2: float = Field(default=0.5, gt=0.0)
    # Douglas-Peucker tolerance on the polygonised mask. One pixel of a 10 m GRD
    # product - enough to drop the staircase, too small to move a boundary.
    detection_simplify_m: float = Field(default=10.0, ge=0.0)

    # ------------------------------- §5.2 oil / look-alike discriminator ----

    discriminator_oil_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    discriminator_top_k_factors: int = Field(default=6, gt=0)

    # --- rule-based fallback -------------------------------------------------
    # Used only when no trained LightGBM model is on disk. Every weight below is
    # a log-odds contribution set from its §5.2 physics rationale; none is fitted
    # to data and none is tuned to reproduce a figure from the spec. A run scored
    # this way reports method="rule_based" and never claims a model_version.

    # Look-alikes outnumber true slicks among SAR dark formations, so the prior
    # sits well below even odds. This is the score a formation carries before any
    # feature is read, and what a partially-observed formation shrinks back
    # towards.
    rule_base_p_oil: float = Field(default=0.25, gt=0.0, lt=1.0)

    # Wind gate - the §5.2 validity condition, and the single strongest term.
    # Below the gate the sea surface itself mimics oil, so a dark formation there
    # is far more likely to be the sea than a slick. Above it slicks disperse
    # below detectability, which is a weaker argument: the formation is real, it
    # is just unlikely to still be oil. In-band earns a small positive - the
    # observation is at least being made under conditions where it means
    # something. Applied as a step, not a ramp, because §5.2 defines the gate as
    # a binary validity flag rather than a continuum.
    rule_w_wind_gate_low: float = Field(default=2.0, ge=0.0)
    rule_w_wind_gate_high: float = Field(default=1.0, ge=0.0)
    rule_w_wind_in_band: float = Field(default=0.4, ge=0.0)

    # Shape complexity P^2/4*pi*A - 1.0 for a circle. Real slicks are sheared by
    # the drift that moved them and read as filamentary and convoluted;
    # low-wind cells and biogenic films stay broad and rounded.
    rule_w_shape_complexity: float = Field(default=1.2, ge=0.0)
    rule_shape_complexity_pivot: float = Field(default=2.5, gt=0.0)
    rule_shape_complexity_scale: float = Field(default=1.5, gt=0.0)

    # Eccentricity. A deliberate discharge while underway lays oil ALONG the
    # track, so it is elongated - the same physics E2 scores in §5.3. A weather
    # or low-wind feature has no such axis.
    rule_w_eccentricity: float = Field(default=0.9, ge=0.0)
    rule_eccentricity_pivot: float = Field(default=0.80, gt=0.0)
    rule_eccentricity_scale: float = Field(default=0.15, gt=0.0)

    # Edge gradient, dB per pixel. Oil damps capillary waves sharply at its
    # boundary; a low-wind cell fades into the surrounding sea.
    rule_w_edge_gradient: float = Field(default=1.0, ge=0.0)
    rule_edge_gradient_pivot_db: float = Field(default=0.8, gt=0.0)
    rule_edge_gradient_scale_db: float = Field(default=0.4, gt=0.0)

    # Damping contrast, dB below background, as a magnitude. Oil sits 3-10 dB
    # down; a marginal look-alike manages 1-3 dB.
    rule_w_contrast: float = Field(default=1.0, ge=0.0)
    rule_contrast_pivot_db: float = Field(default=4.0, gt=0.0)
    rule_contrast_scale_db: float = Field(default=2.0, gt=0.0)

    # Area, scored on log10(km^2) as a band rather than a direction: below the
    # band a formation is speckle, above it the footprint belongs to weather - a
    # wind shadow spans hundreds of km^2, a discharge does not. Centre 5 km^2,
    # half-width 1.3 decades, so the band runs roughly 0.25-100 km^2.
    rule_w_area: float = Field(default=0.5, ge=0.0)
    rule_area_log10_centre: float = Field(default=0.7)
    rule_area_log10_half_width: float = Field(default=1.3, gt=0.0)

    # ------------------------------------------ §5.3 attribution channels ----

    e1_corridor_sigma_m: float = Field(default=500.0, gt=0.0)
    # The corridor kernel is evaluated on a window of this many sigma around the
    # vessel rather than over the whole raster. At 4 sigma the truncated tail
    # carries under 0.01% of the kernel, and the saving is what keeps a frame of
    # several hundred vessels inside the §7 live-stage budget.
    e1_corridor_truncation_sigma: float = Field(default=4.0, gt=0.0)
    e2_sigma_theta_deg: float = Field(default=25.0, gt=0.0)
    e3_tau_median_min: float = Field(default=90.0, gt=0.0)
    e3_tau_sigma: float = Field(default=0.9, gt=0.0)
    e4_gap_coefficient: float = 0.4
    e4_gap_reference_min: float = 30.0
    e4_gap_cap: float = 3.0
    # Subtracted from a measured gap before E4 scores it. This is a property of
    # how AIS reports, not a tuning parameter: a class A transponder underway
    # reports every few seconds, and a feed decimated to a fixed interval makes
    # normal transmission look like a gap. Without the floor every vessel in the
    # frame collects a boost for its own reporting cadence, which inflates every
    # LR uniformly and can carry a marginal candidate across ln(10). E4 detects
    # going dark; it does not detect transmitting.
    e4_nominal_cadence_min: float = Field(default=15.0, ge=0.0)
    # E4 looks for gaps overlapping [t* - h, t* + h]. A discharge is not
    # instantaneous and t* is itself an estimate, so the window is wider than the
    # snapshot interval it is derived from.
    e4_window_half_h: float = Field(default=1.0, gt=0.0)

    # ------------------------------------------------------ §5.4 fusion ----

    fusion_w1: float = 1.0
    fusion_w2: float = 0.7
    fusion_w3: float = 0.5

    prior_tanker: float = 3.0
    prior_bulk_cargo: float = 1.5
    prior_fishing: float = 1.0
    prior_other: float = 1.0
    prior_passenger: float = 0.5
    prior_history_coefficient: float = 0.5

    @property
    def fusion_weights(self) -> tuple[float, float, float]:
        return (self.fusion_w1, self.fusion_w2, self.fusion_w3)

    @property
    def vessel_type_priors(self) -> dict[str, float]:
        return {
            "tanker": self.prior_tanker,
            "cargo": self.prior_bulk_cargo,
            "bulk": self.prior_bulk_cargo,
            "fishing": self.prior_fishing,
            "passenger": self.prior_passenger,
            "other": self.prior_other,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
