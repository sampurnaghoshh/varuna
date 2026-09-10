# CLAUDE.md — VARUNA

> **Read this fully before your first edit in any session.**
> This file is the single source of truth for architecture, math, and boundaries.
> If a request conflicts with this file, say so before acting.

---

## 0. ASSUMPTIONS (correct these if wrong)

| Assumption | Default | Change if |
|---|---|---|
| GPU for training | Colab / Kaggle T4 | No GPU → demote U-Net to P1, promote `classical.py` CFAR detector to primary |
| Primary demo region | Baltic Sea (real S1 + real Danish AIS) | — |
| Team size | **1 (solo)** | Adjust §11 solo working rules |
| Deadline | T+24h from repo init | — |
| Deployment | **P2 — not a phase.** The demo runs locally from `make up`; nothing on the critical path deploys anywhere | — |
| SAR dataset | **Zenodo Part III** (primary — see below) | Unavailable at the T+3h gate → §0 Data Contingency, Tier 2 |

**Dataset — Tier 1, the one we build against.** Zenodo record **`13761290`**, **Part III**, file `02_Test_images_and_ground_truth.7z` (9.9 GB). It contains three class folders — **oil**, **look-alike**, and **oil-free** — each with ground-truth masks. Scenes are georeferenced **VV+VH Sigma0 in dB, 2048 x 2048 x 2**.

**Part I and Part II are not used in this project.** Do not download, reference, cite, or write code paths for them anywhere.

**Data Split — deterministic, enforced in code.** Part III contains **150 images per class** (oil, look-alike, oil-free). The split is a function of the **filename index**, identical for every class:

| Index range | Use |
|---|---|
| `001–120` | train |
| `121–145` | val |
| `146–150` | **HOLDOUT — demo scenes only** |

- The three demo scenes (SC-01 / SC-02 / SC-03, §12) are drawn **only** from the holdout range.
- **No model trains or validates on a holdout index. Ever.** Not for a sanity check, not "just this once", not with a comment explaining why it was fine.
- The split is enforced **programmatically, not by convention**, and it lives in **exactly one place**: `scripts/data_split.py`, exposing `get_split(class_name) -> dict` with `train` / `val` / `holdout` index ranges. `ml/train_segmenter.py` and `ml/train_discriminator.py` **import it** — they never reimplement, inline, hardcode, or "temporarily override" the ranges. Two implementations means one of them is wrong and nobody knows which.
- **Enforcement raises on holdout access.** Not a warning, not a silent filter — a training or validation path that reaches for `146–150` fails loudly and stops.
- **`data/scenes/**` stays read-only and split-unaware.** No enforcement at scene ingest: the demo pipeline is *supposed* to load holdout scenes, and a guard there would either break the demo or teach everyone to bypass the guard. The split is a training-data concern and lives at the training-data loader. See §9.

**Data Contingency.** Zenodo Part III is primary. It is also a 9.9 GB download from a host that has already returned HTTP 504 during this build, so the fallback is written down now rather than improvised against the clock.

| Tier | Source | SAR pixels | Georeferencing | UI badge |
|---|---|---|---|---|
| **1** | **Zenodo Part III** (primary) | real | real | none |
| **2** | **Kaggle Deep-SAR SOS** (fallback) | real | **assigned** over the Baltic AOI | `GEOREFERENCE ASSIGNED — PIXELS ARE REAL SAR` |
| **3** | **Copernicus Data Space direct S1 GRD fetch** | real | real | **OUT OF SCOPE — never attempt** |
| **4** | **Synthetic SAR** | synthetic | synthetic | explicit decision required — see below |

- **Decision point is T+3h.** §15 puts the fixture deadline at T+5h, and fixtures must come from real engine runs on the real scenes — so the dataset has to be settled *well before* that hour, not at it. If Part III has not landed by T+3h, drop to Tier 2 and move on. Do not spend hour 4 retrying a download.
- **The T+3h gate is about code, not switching cost.** Dropping to Tier 2 costs roughly **20 minutes**, not two hours. The gate exists so that ingest is written against the right directory structure, index convention and mask format from the start. Rewriting a loader at T+6h is what costs the day — the switch itself is cheap.
- **Tier 2 pixels are real SAR. The georeferencing is not.** Deep-SAR SOS ships without a geotransform, so we assign one over the Baltic AOI to place the scene in the same frame as the real Danish AIS. Every affected scene carries `GEOREFERENCE ASSIGNED — PIXELS ARE REAL SAR` in the UI, per §2.3, for exactly the same reason injected AIS carries its badge: we win Q&A by being honest.
- **Deep-SAR SOS mask availability is UNVERIFIED.** Confirm the fallback actually ships ground-truth masks **before** committing to Tier 2. If it does not, §12's "bundled ground-truth mask as detection truth" no longer holds and detection truth has to come from somewhere else — that is an explicit decision, not a silent workaround at hour 6.
- **What the Tier 2 badge obliges us to say out loud.** With an assigned geotransform the drift physics and the AIS correlation run over a placed scene, not a surveyed one. A Tier 2 attribution demonstrates that the method works; it is not a claim about a real vessel at a real position. Say that before a judge asks.
- **Tier 4 is a decision, never a silent fallback.** No code path degrades into synthetic SAR on its own. If Tier 4 is ever reached it is because a human chose it explicitly, and the scene is badged as simulated throughout (§2.3). A pipeline that quietly starts inventing its own input is the precise failure §2.2 exists to prevent.
- **The split ranges above are Part III's.** `001–120 / 121–145 / 146–150` describes 150 images per class in that specific dataset. Under Tier 2 those indices mean nothing, and `scripts/data_split.py` must be re-pointed at the fallback's own structure before a single epoch runs — the *discipline* (a demo-only holdout, enforced in code, raising on access) survives the tier change even though the numbers do not.
- **Tier 3 — Copernicus Data Space direct S1 GRD fetch — is EXPLICITLY OUT OF SCOPE for this build.** Real pixels and real georeferencing, at roughly **1 GB per scene** plus manual slick hunting to find a scene that contains a spill at all. That is unaffordable for one developer in 24 hours. It exists in this ladder for completeness and to answer *"why not real live data?"* in Q&A — it is not a reachable option, and it is **never attempted during the build** (§10 P2-15).

---

## 1. WHAT WE ARE BUILDING

**VARUNA** — *Vessel Attribution via Reverse-drift & Unified Nowcast Analytics*

SIH26143 (NTRO) — detect oil spills in satellite SAR, and identify the responsible vessel via AIS correlation.

**One line:** VARUNA rewinds the ocean — it reverse-simulates drift physics from a detected slick to reconstruct where and when the oil entered the water, then scores every AIS-visible vessel against that origin with a forensic likelihood ratio.

**The insight that defines the whole codebase:** the slick in the image is *not* where it was released. It has drifted, stretched, and diffused for hours. Naive spatial matching of AIS tracks to the observed slick footprint accuses innocent vessels. Everything in `backend/app/drift/` and `backend/app/attribution/` exists to fix that.

**We are not building a spill detector.** Detection is stage 1 of 4 and is the least interesting part. Do not gold-plate it.

---

## 2. NON-NEGOTIABLES

1. **The 3-minute demo must run offline, on a laptop, with no internet.** Every architectural choice defers to this.
2. **No fake intelligence.** No hardcoded "AI response" strings. Every number on screen comes from a real model output, a real physics run, or a fixture that was itself generated by one.
3. **Simulated data is labelled as simulated, in the UI, visibly.** Injected AIS tracks render with an `INJECTED — SIMULATED` badge. We win Q&A by being honest, not by hiding.
4. **The system must be able to say "I don't know."** `UNATTRIBUTED` is a first-class output, not an error path.
5. **P0 > P1 > P2, always.** If a P1 feature threatens a P0 feature at any point, delete the P1 feature. Do not negotiate with it.

---

## 3. TECH STACK (do not substitute without asking)

**Backend:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.0, asyncpg, uvicorn
**DB:** PostgreSQL 16 + **PostGIS 3.4**. GIST index on all geometry, BRIN on `ais_positions.ts`.
**ML:** PyTorch, `segmentation-models-pytorch` (U-Net / ResNet34), LightGBM, SHAP, scikit-learn
**Geo/Physics:** NumPy, SciPy, Shapely 2.x, GeoPandas, rasterio, pyproj
**Realtime:** Redis pub/sub → FastAPI WebSocket
**Frontend:** Next.js 14 (App Router), TypeScript strict, Tailwind, shadcn/ui, **deck.gl + MapLibre GL**, Framer Motion, Recharts
**Infra:** Docker Compose

### Explicit non-choices — do not add these
- **No pgvector.** There is no semantic-similarity problem here. Every query is spatiotemporal. Adding it is AI theatre and a judge will catch it.
- **No Mapbox.** MapLibre only — no API token to expire or rate-limit on stage.
- **No auth.** Not relevant to the demo. P2 at best.
- **No chatbot.** Anywhere. For any reason.
- **No LLM except one place:** the narrative paragraph of the evidence dossier, with all numbers injected from pipeline output, never generated, with a Jinja template fallback when the API is unreachable. **The dossier is P2 (§10) — no LLM runs anywhere in this build.**
- **No ORM magic in hot paths.** Raw SQL with PostGIS functions where it matters.

---

## 4. REPOSITORY LAYOUT

```
varuna/
├── CLAUDE.md                      # this file
├── docker-compose.yml
├── Makefile                       # make up / make seed / make test / make demo
├── .env.example
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py              # pydantic-settings, all tunables live here
│   │   ├── db/
│   │   │   ├── models.py          # SQLAlchemy models
│   │   │   ├── session.py
│   │   │   └── schema.sql         # PostGIS DDL, source of truth
│   │   ├── api/v1/
│   │   │   ├── scenes.py  detect.py  drift.py
│   │   │   ├── attribute.py  dossier.py  impact.py
│   │   │   ├── demo.py    ws.py     health.py
│   │   ├── detection/
│   │   │   ├── segmenter.py       # U-Net inference
│   │   │   ├── classical.py       # CFAR/adaptive-threshold fallback — MUST always work
│   │   │   ├── features.py        # geometry + texture feature extraction
│   │   │   ├── discriminator.py   # LightGBM oil vs look-alike
│   │   │   └── explain.py         # SHAP → UI payload
│   │   ├── drift/
│   │   │   ├── solver.py          # RK4 Lagrangian, forward + backward
│   │   │   ├── fields.py          # wind/current field loaders + synthetic fallback
│   │   │   └── density.py         # particle cloud → KDE probability field + hull
│   │   ├── attribution/
│   │   │   ├── channels.py        # E1..E4
│   │   │   ├── priors.py          # vessel type / history priors
│   │   │   └── fusion.py          # Bayesian LR combination + decision bands
│   │   ├── ais/
│   │   │   ├── decoder.py         # AIVDM + Danish CSV
│   │   │   ├── ingest.py
│   │   │   └── tracks.py          # track building + gap detection
│   │   ├── dossier/
│   │   │   ├── builder.py         # MARPOL Annex I PDF + SHA-256
│   │   │   └── templates/
│   │   ├── schemas/               # Pydantic request/response models ONLY
│   │   └── fallback/
│   │       └── loader.py          # serves data/fixtures/ when live path fails
│   ├── ml/
│   │   ├── train_segmenter.py
│   │   ├── train_discriminator.py
│   │   ├── eval_detection.py
│   │   └── eval_attribution.py    # synthetic-truth attribution benchmark
│   └── tests/
│
├── frontend/
│   ├── app/                       # App Router
│   ├── components/
│   │   ├── map/                   # deck.gl layers
│   │   ├── panels/                # detection, explainability, attribution
│   │   ├── demo/                  # demo mode controller + scrubber
│   │   └── ui/                    # shadcn primitives
│   ├── lib/                       # api client, ws client, formatters
│   └── types/                     # mirrors backend/app/schemas — keep in sync
│
├── data/
│   ├── scenes/                    # bundled Sentinel-1 GeoTIFFs
│   ├── ais/                       # Danish AIS subset (CSV)
│   ├── models/                    # trained weights
│   └── fixtures/                  # ⛔ FROZEN — see §9
└── scripts/
    ├── download_data.sh
    ├── data_split.py               # get_split(class) — THE split, imported by ml/, never copied
    ├── seed_db.py
    └── build_fixtures.py
```

---

## 5. THE MATH — implement exactly as specified

Do not invent your own formulation. If you think a formula is wrong, raise it, don't silently change it.

### 5.1 Reverse-drift solver (`drift/solver.py`)

Backward Lagrangian integration, RK4, per particle:

```
v(x, t) = u_current(x, t) + α · u_wind10(x, t) + u_stokes(x, t)
```

- `α` (windage) = **0.033** default, exposed as a tunable in range [0.020, 0.040]
- Wind deflection angle `θ_dev` = **0°** default, exposed in [-20°, +20°] (Coriolis-driven veer)
- Timestep `dt` = **300 s**; horizon **12 h**; **5000 particles**
- Backward mode integrates `-v`, forward mode integrates `+v`
- Turbulent diffusion added each step in **both** modes (it is symmetric):
  ```
  Δx_diff = N(0, σ),  σ = sqrt(2 · K_h · dt),  K_h = 10 m²/s default
  ```
- Particles seeded by area-weighted uniform sampling inside the detected slick polygon
- Emit a state snapshot every **30 min** of model time → this is what the UI scrubber consumes

Output per snapshot: particle positions, an alpha-hull uncertainty polygon, and a KDE probability field `O(x, y, t)` normalised to sum to 1 across the whole run.

**Field sourcing (`drift/fields.py`)** — try in order, log which was used:
1. Cached CMEMS/ERA5 NetCDF in `data/` (preferred)
2. Bundled per-scenario field snapshot
3. **Synthetic field**: geostrophic-like flow with 2–3 mesoscale eddies + spatially coherent wind. Must be deterministic given a seed.

Never let a missing field crash the run. Degrade and set `field_source` in the response.

### 5.2 Detection features (`detection/features.py`)

Extract per dark-formation polygon:

`area_km2`, `perimeter_km`, `shape_complexity` (P²/4πA), `major_axis_km`, `minor_axis_km`, `eccentricity`, `orientation_deg`, `mean_sigma0_db`, `std_sigma0_db`, `contrast_db` (slick mean − local background mean), `edge_gradient_mean`, `edge_gradient_std`, `glcm_homogeneity`, `glcm_contrast`, `glcm_entropy`, `wind_speed_ms`, `distance_to_coast_km`, `n_ships_within_20km`

**Wind gate:** if `wind_speed_ms < 3.0` or `> 12.0`, set `wind_gate_violated = True`. Below 3 m/s the sea surface itself mimics oil; above 12 m/s slicks disperse below detectability. This flag must surface in the UI explainability panel — it is one of our strongest Q&A answers.

### 5.3 Attribution channels (`attribution/channels.py`)

For each candidate vessel `j` with AIS presence intersecting the origin cone:

**E1 — spatiotemporal mass overlap**
```
m_j = Σ_t ∫ O(x, y, t) · C_j(x, y, t) dA
```
`C_j` = Gaussian corridor kernel, σ_c = 500 m, around vessel `j`'s interpolated position at time `t`.
`s1_j = m_j / Σ_k m_k`. Record `t*_j` = the snapshot time contributing maximum mass.

**E2 — axial coherence** *(independent of the current field)*
```
Δθ = fold(|orientation_slick − COG_j(t*_j)|, 90°)
s2_j = exp(−(Δθ / σ_θ)²),   σ_θ = 25°
```
Rationale: deliberate discharge while underway lays oil *along* the track.

**E3 — kinematic consistency** *(independent of the current field)*
```
L_released = major_axis_km / stretch_factor      # stretch_factor from the drift run
τ_j        = L_released / SOG_j                  # implied discharge duration
s3_j       = lognormal_pdf(τ_j; median = 90 min, σ = 0.9) normalised to peak 1.0
```

**E4 — dark-gap coincidence**
```
g_j  = longest AIS gap (minutes) overlapping [t*_j − 1h, t*_j + 1h]
s4_j = 1 + 0.4 · min(g_j / 30, 3)     # boost only, max 2.2×; never penalise clean transmitters
```

E2 and E3 must not touch the current field. That independence is the entire reason the system survives a coarse ocean model, and it is our answer to the sharpest judge question. Do not refactor it away.

### 5.4 Fusion (`attribution/fusion.py`)

Background terms `s_bg` are the mean channel scores across all vessels in the frame. This is what makes the output a likelihood *ratio* rather than an arbitrary score.

```
log LR_j = w1·log(s1_j/s1_bg) + w2·log(s2_j/s2_bg) + w3·log(s3_j/s3_bg)
           + log(s4_j) + log(π_j)

w = [1.0, 0.7, 0.5]
```

**Priors `π_j`** (`priors.py`), normalised: tanker 3.0 · bulk/cargo 1.5 · fishing 1.0 · other 1.0 · passenger 0.5, multiplied by `(1 + 0.5 · prior_detections_j)`.

**Decision bands — hard-coded, never bypassed:**

| log LR | LR | Verdict |
|---|---|---|
| ≥ ln(100) | ≥ 100 | `STRONG` |
| ≥ ln(10) | 10–100 | `MODERATE` |
| < ln(10) | < 10 | `UNATTRIBUTED` |

`UNATTRIBUTED` returns a queued case with a cross-check recommendation. **It never returns the top-ranked vessel as a culprit.** No code path may promote a sub-threshold candidate to an accusation.

Posterior = softmax over `log LR_j` including an explicit "none of the above" hypothesis pinned at `ln(10)`.

---

## 6. DATABASE SCHEMA (`db/schema.sql` is authoritative)

```sql
scenes(id, product_id, sensor, acq_time timestamptz, footprint geometry(Polygon,4326),
       incidence_angle_deg, wind_speed_ms, wind_dir_deg, pixel_spacing_m, source, raster_path)

detections(id, scene_id FK, geom geometry(Polygon,4326), <all §5.2 features>,
           p_oil real, class text, wind_gate_violated bool, model_version, detector text)

detection_factors(detection_id FK, feature text, value real, shap real)

drift_runs(id, detection_id FK, mode text, n_particles int, horizon_h int,
           windage real, k_h real, field_source text, params jsonb, created_at)

drift_states(run_id FK, t_offset_min int, particles geometry(MultiPoint,4326),
             hull geometry(Polygon,4326), density jsonb, stretch_factor real)

vessels(mmsi bigint PK, imo, name, callsign, type, length_m, width_m, flag,
        prior_detections int default 0, is_injected bool default false)

ais_positions(id, mmsi FK, ts timestamptz, geom geometry(Point,4326),
              sog real, cog real, heading real, nav_status, is_injected bool)
    -- GIST(geom), BRIN(ts), INDEX(mmsi, ts)

ais_tracks(id, mmsi FK, t_start, t_end, path geometry(LineString,4326),
           gap_count int, max_gap_min real)

attributions(id, detection_id FK, mmsi FK, rank int, log_lr real, lr real,
             posterior real, verdict text, channels jsonb, t_star timestamptz, created_at)

dossiers(id, detection_id FK, mmsi FK, pdf_path, sha256, officer_ack bool default false, generated_at)

demo_scenarios(id, code text unique, name, config jsonb)
```

---

## 7. API CONTRACT

All under `/api/v1`. Every response includes `{ "source": "live" | "fixture", "elapsed_ms": int }`.

```
GET    /health
GET    /scenes                              list bundled scenes
POST   /scenes/ingest                       upload GeoTIFF
GET    /scenes/{id}

POST   /detect/{scene_id}                   → detections + p_oil + SHAP factors
GET    /detections/{id}

POST   /drift/backward                      {detection_id, horizon_h, windage, k_h, theta_dev}
POST   /drift/forward                       {detection_id, horizon_h}
GET    /drift/{run_id}/frames?step_min=30   → scrubber frames

POST   /attribute/{detection_id}            → ranked candidates + channel breakdown + verdict
GET    /vessels/{mmsi}/track?from&to

GET    /impact/{detection_id}               forward-drift landfall ETA + exposure
POST   /dossier/{detection_id}/{mmsi}       → PDF + sha256

POST   /demo/run/{scenario_code}            fires WS pipeline events
POST   /demo/reset

WS     /ws/pipeline/{job_id}
```

**WebSocket event shape** — the UI depends on this exactly:
```json
{ "stage": "SEGMENTING" | "DISCRIMINATING" | "REWINDING" | "FUSING" | "DONE" | "ERROR",
  "progress": 0.0-1.0, "message": "human readable", "payload": {} }
```
Stage messages are what make the judge watch the system *think*. Write them well: `"Rewinding ocean state to T−4h 30m"`, not `"Processing..."`.

**P2 endpoints — `GET /impact/{detection_id}` and `POST /dossier/{detection_id}/{mmsi}`.** These stay in the contract but are not implemented in this build (§10). They return **HTTP 501** with a clear typed message — e.g. `{"detail": "Forward impact is P2 and not implemented in this build."}` — never a 404, never an empty 200, never a plausible-looking stub payload. §2.2 forbids the stub.

**No dead buttons.** The frontend must not render *any* control that calls a P2 endpoint: no greyed-out "Generate dossier", no disabled impact tab, no "coming soon" tooltip. The control does not exist. A judge clicking something that does nothing costs more than the absent feature ever would.

---

## 8. CONVENTIONS

- **Python:** full type hints, `ruff` clean, Pydantic v2 for every boundary, no bare `except`. Tunables in `config.py`, never inline literals.
- **TypeScript:** `strict: true`, no `any`, types in `frontend/types/` mirroring `backend/app/schemas/`.
- **Units are in names.** `distance_km`, `wind_speed_ms`, `t_offset_min`. Never a bare `distance`.
- **CRS:** everything stored EPSG:4326; project to a local UTM zone for metric computation, never compute distances in degrees.
- **Errors:** every stage returns a typed error and degrades to the fixture path. Nothing raises to the user.
- **Comments:** only for physics/statistics rationale. Do not comment obvious code.
- **Commits:** small, scoped to one module, imperative subject line.

---

## 9. ⛔ DO NOT TOUCH

| Path | Rule |
|---|---|
| `data/fixtures/**` | **FROZEN once green.** These are the offline demo fallbacks. Never regenerate, never "improve", never let a refactor rewrite them. If a fixture is genuinely wrong, say so and stop — a human decides. |
| `data/scenes/**`, `data/ais/**` | Read-only inputs. Never modify in place. |
| `app/attribution/fusion.py` decision bands | The 10 / 100 thresholds are a safety property, not a tuning parameter. |
| `app/detection/classical.py` | The always-works fallback. Keep it dependency-light and never let it import torch. |
| Part III indices **146–150**, every class | **HOLDOUT.** Excluded from all training and validation code paths, permanently (§0 Data Split). These are the demo scenes — a model that has seen them makes the demo a lie, and the enforcement in `scripts/` is not a check to be relaxed when a run is inconvenient. Under §0 Data Contingency Tier 2 these specific indices no longer apply, but the holdout discipline does: re-point `scripts/data_split.py` at the fallback's structure, never remove the guard. |

**After T+14h:** no new dependencies, no refactors, no renames. Bug fixes and polish only.

---

## 10. FEATURE PRIORITY

**P0 — the demo dies without these**
1. Scene ingest, 3 bundled Part III scenes — one per class, drawn from the **holdout range `146–150`** (§0 Data Split). Under §0 Data Contingency Tier 2 the scenes come from the fallback instead and carry the georeference badge.
2. Segmentation → polygons + features — trained on `001–120`, validated on `121–145`, **never** on holdout
3. LightGBM discriminator + SHAP panel — same split, same enforcement
4. Reverse-drift solver + frames API
5. AIS ingest → PostGIS tracks + gaps
6. LR attribution + channel breakdown
7. deck.gl console: slick, particle rewind, tracks, culprit highlight
8. Demo Mode, 3 scenarios, WS-driven, under 3 min
9. Fixture fallback for every scenario

**P1 — wow, only after all of P0 is green**
10. What-if sliders · 11. **Precomputed windage variants** · 12. Dark-vessel alert

**P2 — almost certainly not**
13. Evidence dossier PDF · 14. Forward impact **(incl. impact metrics)** · 15. Live Copernicus fetch — **out of scope, see §0 Tier 3** · 16. LLM narrative · 17. Multi-pass tracking · 18. Auth · 19. Deployment

---

## 11. SOLO WORKING RULES

One developer, one session at a time. There is no parallel work, no worktree split, and no shared-file announcement protocol — that coordination cost no longer exists.

1. **One session per module.** A session works `drift/` *or* `detection/` *or* `frontend/**` — never two at once. Finish the module and hand back before switching.
2. **Re-read CLAUDE.md at the start of every session.** Fully, before the first edit. This file changes between sessions; your memory of it does not.
3. **Plan mode is mandatory for `drift/` and `attribution/`.** No exceptions, no "it's a small change". Those two modules carry the whole project (§14).
4. **`make test` before every handback.** Green, or it is not handed back.

---

## 12. DEMO SCENARIOS (`data/fixtures/`, driven by `/demo/run/{code}`)

All scenes come from **Zenodo Part III** (record `13761290`, `02_Test_images_and_ground_truth.7z`, §0) — Tier 1. Each scenario draws its scene from the **holdout range of its matching class folder** — indices `146–150`, never seen by any model (§0 Data Split, §9) — using the bundled ground-truth mask as detection truth.

Under **§0 Data Contingency Tier 2** the scenes come from Deep-SAR SOS instead, every one of them badged `GEOREFERENCE ASSIGNED — PIXELS ARE REAL SAR`, and the holdout is reconstituted against that dataset's own structure. The detection-truth claim above depends on the fallback shipping masks, which is **UNVERIFIED** — see §0.

**SC-01 · "Baltic Night Discharge"** — 110 s. Scene from **`oil/146–150`**, real Danish AIS traffic, one injected culprit track (badged `INJECTED — SIMULATED`). Full pipeline → particle rewind → culprit ignites at ~T−6h20m → `STRONG`, LR ≈ 4000+.

**SC-02 · "The Look-alike Trap"** — **P0-CRITICAL** — 45 s. Scene from **`look-alike/146–150`**, wind 2.1 m/s. Returns `LOOK-ALIKE`, P(oil) ≈ 0.12, wind gate violated, **no attribution issued**. This scenario is as important as SC-01 — it proves the system refuses to accuse. It ships, or the demo does not run.

**SC-03 · "Clean Sea"** — 15 s. Scene from **`oil-free/146–150`**. Zero detections. Proves no false positives.

Total ≤ 170 s. Rehearse twice before the demo. If the run goes long, trim SC-03 — never SC-01, never SC-02.

---

## 13. COMMANDS

```bash
make up          # docker compose up --build (db, redis, backend)
make reset-db    # DESTROY the db volume and re-apply schema.sql
make seed        # schema + scenes + AIS + demo scenarios
make test        # pytest + tsc --noEmit
make demo        # run all 3 scenarios headless, assert expected verdicts
make fixtures    # regenerate data/fixtures — HUMAN-INVOKED ONLY
```

`make demo` is the release gate. It must pass before every rehearsal and before sleeping.

**Schema edits require `make reset-db`.** `backend/app/db/schema.sql` is executed by the postgis container from `/docker-entrypoint-initdb.d`, which runs **only on an empty volume**. Editing `schema.sql` and restarting does nothing — the column never appears and the failure is silent. `make reset-db` is `docker compose down -v && docker compose up -d db`. It destroys all data in the database; re-run `make seed` afterwards.

**The frontend is not containerised.** Compose runs postgres, redis and the backend only. Run the UI natively — `cd frontend && npm run dev` (or `make frontend`) — against `http://localhost:8000`. Next dev-server hot reload over Windows bind mounts is too slow and unreliable for the hours spent in `frontend/**`.

---

## 14. HOW TO WORK IN THIS REPO

- **Use Plan mode for `drift/` and `attribution/`.** Those two modules carry the whole project. A wrong assumption there costs hours. Everywhere else, move fast.
- **Run `make test` yourself** after every change. Do not hand back unverified code.
- **Verify, don't assume.** If a file's state matters, read it.
- **Surface disagreement early.** If something in this file looks wrong, say so before writing code around it.
- **When time is short, delete scope — never quality of the P0 path.** A smaller demo that runs beats a larger one that stalls in front of a judge.

---

## 15. FIXTURE-FIRST BUILD ORDER

`data/fixtures/` is not a fallback bolted on at the end. It is the **contract the frontend is built against**, and it exists before the frontend does.

**Hour 5 — generate the fixtures.** By T+5h the drift solver (§5.1) and the attribution engine (§5.3–5.4) must run end-to-end on all three Part III scenes — well enough to emit real output, not polished. `scripts/build_fixtures.py` captures that output into `data/fixtures/`: detections + features + SHAP factors, drift frames at 30-min steps, ranked attributions with channel breakdowns, and the full WS event sequence per scenario. Every number in there is a genuine engine output — §2.2 applies here above all.

**Hour 5 onward — the entire frontend is developed against those fixtures.** No live backend wiring until the frontend is complete against the fixture set. The API client reads `data/fixtures/` through `app/fallback/loader.py`; `"source": "fixture"` (§7) is the normal case during development, not an error state.

**Why this ordering, not the obvious one:**
- The frontend never blocks on backend availability, and backend churn never breaks the UI mid-build.
- The offline demo path (§2.1) is exercised from hour 5, not discovered at hour 22.
- Fixtures freeze the schema early. If a fixture shape has to change, a Pydantic schema was wrong — and hour 5 is when you want to learn that.

Once green, fixtures are **FROZEN** (§9). `make fixtures` regenerates them and is human-invoked only.

---

## 16. REPORTING NUMBERS

- **Quote detection metrics from val — `121–145`, n = 25 per class — and say so.** Every figure carries its split name and its n: "F1 0.87 (val, n=25)", never a bare "F1 0.87". Those ranges and that n are Part III's; under §0 Data Contingency Tier 2 both change, and the figure carries whatever the fallback's split and n actually are.
- **State that hyperparameters were selected on val.** They were. That makes val an optimistic estimate, and saying it out loud costs nothing and buys the room's trust. A judge who extracts that fact from you has won a point; a judge who hears it from you first has not.
- **Never quote a metric from holdout.** `146–150` is n = 5 per class — a number computed on it is noise with a decimal point. Holdout exists for exactly one reason: **demo-scene provenance**, the proof that no model has seen SC-01/02/03 (§0, §9). It is not a test set and we do not report it as one.
- **Any accuracy figure in the UI or the pitch deck carries its split and n.** No exceptions. An unlabelled number on screen is the kind of thing §2.2 exists to prevent.
- If a number cannot be traced to a split, an n, and a run, it does not go on screen.
