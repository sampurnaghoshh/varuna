-- VARUNA — PostGIS DDL. Source of truth for §6 of CLAUDE.md.
--
-- Executed by the postgis container from /docker-entrypoint-initdb.d ON AN EMPTY
-- VOLUME ONLY. Editing this file does not migrate a running database:
-- run `make reset-db` (docker compose down -v) to apply any change made here.
--
-- CRS is EPSG:4326 everywhere (§8). Metric computation projects to a local UTM
-- zone in application code; nothing computes distance in degrees.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------- scenes ----

CREATE TABLE IF NOT EXISTS scenes (
    id                    bigserial PRIMARY KEY,
    product_id            text NOT NULL UNIQUE,
    sensor                text NOT NULL,
    acq_time              timestamptz NOT NULL,
    footprint             geometry(Polygon, 4326) NOT NULL,
    incidence_angle_deg   real,
    wind_speed_ms         real,
    wind_dir_deg          real,
    pixel_spacing_m       real,
    source                text,
    raster_path           text NOT NULL
);

CREATE INDEX IF NOT EXISTS scenes_footprint_gist ON scenes USING GIST (footprint);
CREATE INDEX IF NOT EXISTS scenes_acq_time_idx   ON scenes (acq_time);

-- ------------------------------------------------------------ detections ----
-- Feature columns are the §5.2 set, expanded one column per feature rather than
-- a jsonb blob: the discriminator and the SHAP panel both address them by name.

CREATE TABLE IF NOT EXISTS detections (
    id                    bigserial PRIMARY KEY,
    scene_id              bigint NOT NULL REFERENCES scenes (id) ON DELETE CASCADE,
    geom                  geometry(Polygon, 4326) NOT NULL,

    area_km2              real,
    perimeter_km          real,
    shape_complexity      real,
    major_axis_km         real,
    minor_axis_km         real,
    eccentricity          real,
    orientation_deg       real,
    mean_sigma0_db        real,
    std_sigma0_db         real,
    contrast_db           real,
    edge_gradient_mean    real,
    edge_gradient_std     real,
    glcm_homogeneity      real,
    glcm_contrast         real,
    glcm_entropy          real,
    wind_speed_ms         real,
    distance_to_coast_km  real,
    n_ships_within_20km   int,

    p_oil                 real,
    class                 text CHECK (class IN ('oil', 'look-alike', 'oil-free')),
    wind_gate_violated    boolean NOT NULL DEFAULT false,
    model_version         text,
    detector              text
);

CREATE INDEX IF NOT EXISTS detections_geom_gist ON detections USING GIST (geom);
CREATE INDEX IF NOT EXISTS detections_scene_idx ON detections (scene_id);

-- ----------------------------------------------------- detection_factors ----
-- One row per feature per detection: the contribution behind the explainability
-- panel (§10 P0-3).
--
-- `shap` holds the contribution whichever way it was produced, and `basis` says
-- which way that was. When no trained discriminator is available the fallback is
-- a rule-based scorer over the §5.2 physics, and its contributions are additive
-- log-odds in exactly the same units - identical in the column, and not the same
-- claim. There is deliberately NO DEFAULT: defaulting to 'shap' would relabel
-- every rule-based row as a model output, which is the precise confusion the
-- column exists to prevent, so a writer has to state which it is.

CREATE TABLE IF NOT EXISTS detection_factors (
    id                    bigserial PRIMARY KEY,
    detection_id          bigint NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    feature               text NOT NULL,
    value                 real,
    shap                  real,
    basis                 text NOT NULL CHECK (basis IN ('shap', 'rule_based')),
    UNIQUE (detection_id, feature)
);

CREATE INDEX IF NOT EXISTS detection_factors_detection_idx ON detection_factors (detection_id);

-- ------------------------------------------------------------ drift_runs ----

CREATE TABLE IF NOT EXISTS drift_runs (
    id                    bigserial PRIMARY KEY,
    detection_id          bigint NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    mode                  text NOT NULL CHECK (mode IN ('forward', 'backward')),
    n_particles           int NOT NULL,
    horizon_h             int NOT NULL,
    windage               real NOT NULL,
    k_h                   real NOT NULL,
    field_source          text NOT NULL,
    params                jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at            timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS drift_runs_detection_idx ON drift_runs (detection_id);

-- ---------------------------------------------------------- drift_states ----
-- One row per 30-min model-time snapshot (§5.1). `density` holds the KDE
-- probability field O(x,y,t); `stretch_factor` feeds attribution channel E3.

CREATE TABLE IF NOT EXISTS drift_states (
    id                    bigserial PRIMARY KEY,
    run_id                bigint NOT NULL REFERENCES drift_runs (id) ON DELETE CASCADE,
    t_offset_min          int NOT NULL,
    particles             geometry(MultiPoint, 4326),
    hull                  geometry(Polygon, 4326),
    density               jsonb,
    stretch_factor        real,
    UNIQUE (run_id, t_offset_min)
);

CREATE INDEX IF NOT EXISTS drift_states_particles_gist ON drift_states USING GIST (particles);
CREATE INDEX IF NOT EXISTS drift_states_hull_gist      ON drift_states USING GIST (hull);
CREATE INDEX IF NOT EXISTS drift_states_run_idx        ON drift_states (run_id, t_offset_min);

-- --------------------------------------------------------------- vessels ----

CREATE TABLE IF NOT EXISTS vessels (
    mmsi                  bigint PRIMARY KEY,
    imo                   bigint,
    name                  text,
    callsign              text,
    type                  text,
    length_m              real,
    width_m               real,
    flag                  text,
    prior_detections      int NOT NULL DEFAULT 0,
    is_injected           boolean NOT NULL DEFAULT false
);

-- --------------------------------------------------------- ais_positions ----
-- The hot table. BRIN on ts because rows land in near-timestamp order and the
-- table is large; GIST on geom for the corridor-kernel lookups in E1.

CREATE TABLE IF NOT EXISTS ais_positions (
    id                    bigserial PRIMARY KEY,
    mmsi                  bigint NOT NULL REFERENCES vessels (mmsi) ON DELETE CASCADE,
    ts                    timestamptz NOT NULL,
    geom                  geometry(Point, 4326) NOT NULL,
    sog                   real,
    cog                   real,
    heading               real,
    nav_status            text,
    is_injected           boolean NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS ais_positions_geom_gist ON ais_positions USING GIST (geom);
CREATE INDEX IF NOT EXISTS ais_positions_ts_brin   ON ais_positions USING BRIN (ts);
CREATE INDEX IF NOT EXISTS ais_positions_mmsi_ts   ON ais_positions (mmsi, ts);

-- ------------------------------------------------------------ ais_tracks ----

CREATE TABLE IF NOT EXISTS ais_tracks (
    id                    bigserial PRIMARY KEY,
    mmsi                  bigint NOT NULL REFERENCES vessels (mmsi) ON DELETE CASCADE,
    t_start               timestamptz NOT NULL,
    t_end                 timestamptz NOT NULL,
    path                  geometry(LineString, 4326) NOT NULL,
    gap_count             int NOT NULL DEFAULT 0,
    max_gap_min           real
);

CREATE INDEX IF NOT EXISTS ais_tracks_path_gist ON ais_tracks USING GIST (path);
CREATE INDEX IF NOT EXISTS ais_tracks_mmsi_idx  ON ais_tracks (mmsi, t_start);

-- ---------------------------------------------------------- attributions ----
-- verdict is constrained to the §5.4 decision bands. UNATTRIBUTED is a
-- first-class result (§2.4), not an error state.

CREATE TABLE IF NOT EXISTS attributions (
    id                    bigserial PRIMARY KEY,
    detection_id          bigint NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    mmsi                  bigint NOT NULL REFERENCES vessels (mmsi) ON DELETE CASCADE,
    rank                  int NOT NULL,
    log_lr                real NOT NULL,
    lr                    real NOT NULL,
    posterior             real,
    verdict               text NOT NULL CHECK (verdict IN ('STRONG', 'MODERATE', 'UNATTRIBUTED')),
    channels              jsonb NOT NULL DEFAULT '{}'::jsonb,
    t_star                timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now(),
    UNIQUE (detection_id, mmsi)
);

CREATE INDEX IF NOT EXISTS attributions_detection_rank_idx ON attributions (detection_id, rank);

-- -------------------------------------------------------------- dossiers ----

CREATE TABLE IF NOT EXISTS dossiers (
    id                    bigserial PRIMARY KEY,
    detection_id          bigint NOT NULL REFERENCES detections (id) ON DELETE CASCADE,
    mmsi                  bigint NOT NULL REFERENCES vessels (mmsi) ON DELETE CASCADE,
    pdf_path              text NOT NULL,
    sha256                text NOT NULL,
    officer_ack           boolean NOT NULL DEFAULT false,
    generated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dossiers_detection_idx ON dossiers (detection_id);

-- -------------------------------------------------------- demo_scenarios ----

CREATE TABLE IF NOT EXISTS demo_scenarios (
    id                    bigserial PRIMARY KEY,
    code                  text NOT NULL UNIQUE,
    name                  text NOT NULL,
    config                jsonb NOT NULL DEFAULT '{}'::jsonb
);
