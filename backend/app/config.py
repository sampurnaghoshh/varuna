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
