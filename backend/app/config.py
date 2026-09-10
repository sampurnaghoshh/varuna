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

    # ------------------------------------------- §5.2 detection wind gate ----
    # Below 3 m/s the sea surface itself mimics oil; above 12 m/s slicks
    # disperse below detectability.

    wind_gate_min_ms: float = 3.0
    wind_gate_max_ms: float = 12.0

    # ------------------------------------------ §5.3 attribution channels ----

    e1_corridor_sigma_m: float = Field(default=500.0, gt=0.0)
    e2_sigma_theta_deg: float = Field(default=25.0, gt=0.0)
    e3_tau_median_min: float = Field(default=90.0, gt=0.0)
    e3_tau_sigma: float = Field(default=0.9, gt=0.0)
    e4_gap_coefficient: float = 0.4
    e4_gap_reference_min: float = 30.0
    e4_gap_cap: float = 3.0

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
