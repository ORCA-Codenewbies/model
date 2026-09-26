# ORCA — Dataset & Model Specification

### Priority Agents: PFZ Intelligence · Ocean Analytics · Weather Intelligence · Marine Risk Assessment

---

## Shared Foundation — The Spatio-Temporal Join Key

Every model below is built on the same joinable schema, so results from one model can feed another without reconciliation work:

```text
KEY = (timestamp, latitude, longitude)
```

All four datasets are resampled/interpolated onto this common key before training. This is the fact worth stating explicitly to judges — it shows the "agents talk to each other" story is backed by a real data design, not just API calls between black boxes.

---

## 1. PFZ Intelligence Agent

| Field | Detail |
|---|---|
| **Question it answers** | "Where are the Potential Fishing Zones right now / tomorrow?" |
| **Input variables** | Chlorophyll-a concentration, SST, SST gradient (fronts), surface current speed/direction, mixed-layer depth, day-of-year (season), distance from coast |
| **Dataset source** | INCOIS PFZ advisories (historical, 2003–present) as labels; MOSDAC Oceansat-3 OCM + SST as features; INCOIS OSF for currents/MLD |
| **Target / output** | Binary or graded suitability surface: `PFZ_present ∈ {0,1}` or continuous `PFZ_score ∈ [0,1]` per grid cell |
| **Proposed model** | CNN–LSTM: CNN encodes the spatial chlorophyll/SST raster for a day, LSTM captures how the front has evolved over the previous 5–7 days |
| **Why this model** | PFZ formation is driven by moving thermal/chlorophyll fronts — a static classifier ignores that fronts *travel*; CNN-LSTM is the standard architecture for spatial fields that evolve in time |
| **Fallback if compute-limited** | XGBoost on engineered features (SST gradient magnitude, chlorophyll anomaly, distance to nearest historical PFZ) — much cheaper to defend and still credible |
| **Evaluation** | Precision/Recall/F1 against held-out historical PFZ advisories; spatial IoU between predicted and actual PFZ polygons |

---

## 2. Ocean Analytics Agent

| Field | Detail |
|---|---|
| **Question it answers** | "How suitable/productive is this ocean region right now?" (general-purpose scoring used by other agents, not fishing-specific) |
| **Input variables** | SST, chlorophyll, surface current, MLD, D20 (thermocline depth), wave height, wind, season, location |
| **Dataset source** | INCOIS ROMS (SST/MLD/D20/currents, 2010–present) + MOSDAC Oceansat-3 + Copernicus Marine for cross-validation |
| **Target / output** | Continuous **Ocean Suitability Score** ∈ [0,1], used as an upstream feature by the PFZ and Productivity agents |
| **Proposed model** | Gradient-boosted regression (XGBoost/LightGBM) over engineered oceanographic features |
| **Why this model** | This is a tabular, feature-rich, moderate-data problem — tree ensembles outperform deep nets here and are far easier to explain/interpret (feature importance) to judges |
| **Evaluation** | RMSE/MAE against a held-out set; SHAP values to show which variables drive the score, since judges will ask "why is this region suitable?" |

---

## 3. Weather Intelligence Agent

| Field | Detail |
|---|---|
| **Question it answers** | "Is the weather safe for a fishing trip in this window?" |
| **Input variables** | Wind speed/direction, rainfall, pressure, temperature, humidity, visibility, lightning flag, cyclone distance/category |
| **Dataset source** | Copernicus ERA5 (hourly, 1940–present) for historical training; IMD APIs (cyclone track, wind warnings, lightning, radar) for India-specific operational signals |
| **Target / output** | `weather_risk ∈ [0,1]` or categorical `{NORMAL, CAUTION, DANGEROUS}` |
| **Proposed model** | MLP or XGBoost classifier — simple, well-calibrated, fast to run per-request |
| **Why this model** | This is a bounded, well-labelled meteorological classification problem; a deep sequence model adds complexity without clear benefit unless you're forecasting rather than nowcasting |
| **Evaluation** | Accuracy/F1 per class, calibration curve (predicted probability vs. observed frequency) — calibration matters more than raw accuracy here since this score feeds a safety decision |

---

## 4. Marine Risk Assessment Agent

| Field | Detail |
|---|---|
| **Question it answers** | "Is it safe to fish here, right now, for this vessel?" — the aggregator agent |
| **Input variables** | Outputs of Weather Intelligence + Ocean Analytics (as features, not raw data) + wave height/period + current speed + lightning + cyclone distance + geofence flag + vessel type/length |
| **Dataset source** | Self-constructed: join of the Weather, Ocean, and Geospatial datasets above, with labels created from historical marine-incident/advisory records where available, or expert-defined thresholds where not |
| **Target / output** | `risk_score ∈ [0,100]` and classification `{SAFE, CAUTION, DANGER}` |
| **Proposed model** | XGBoost/MLP trained on the *outputs* of the other three models, not raw sensor data — this is a deliberate architectural choice |
| **Why this model** | Keeps prediction and language generation separated: the Risk model produces the number, the LLM only explains it. This is the answer you give when a judge asks "does the LLM decide if it's safe?" — no, a trained model does; the LLM narrates the reasoning behind an already-computed score |
| **Evaluation** | Precision/Recall/F1 on DANGER class specifically (false negatives here are the costly failure mode) — worth stating this explicitly, it signals safety-conscious ML thinking |

---

## 5. Geospatial Reasoning Agent

| Field | Detail |
|---|---|
| **Question it answers** | "Which PFZ is nearest?" / "Is this point inside a restricted zone?" / "Does this route cross a protected area?" |
| **Input variables** | User coordinate, PFZ point/polygon layer, maritime boundary layer, marine protected area layer, port/harbour layer |
| **Dataset source** | Self-assembled GIS knowledge layer: Indian maritime boundary, MPA/restricted-zone shapefiles, fishing harbour locations, INCOIS PFZ point advisories |
| **Target / output** | Deterministic spatial answers: nearest-neighbour distance, boolean intersection, containment |
| **Proposed model** | **None — no ML.** PostGIS / GeoPandas / Shapely with spatial indexing (`ST_DWithin`, `ST_Intersects`) |
| **Why this model** | These are exact geometric questions with a single correct answer. A trained model would be strictly worse — slower, probabilistic where the ground truth is deterministic, and harder to audit |
| **Evaluation** | Correctness against known test coordinates (unit tests, not ML metrics) — query latency is the more relevant number to report |

---

## 6. Route Optimization Agent

| Field | Detail |
|---|---|
| **Question it answers** | "What's the safest route from A to B?" |
| **Input variables** | Start/end coordinates, a navigable graph over the sea (grid or waypoint graph), and per-edge costs pulled from the Weather, Ocean, and Geospatial agents |
| **Dataset source** | No standalone dataset — this agent consumes the *outputs* of Weather Intelligence, Marine Risk Assessment, and Geospatial Reasoning as edge weights |
| **Target / output** | An ordered waypoint list minimizing a combined cost function |
| **Proposed model** | **None — no ML.** A* or Dijkstra over a cost graph where `edge_cost = distance + wave_risk + wind_risk + current_risk + hazard_penalty + geofence_penalty` |
| **Why this model** | Routing is a well-posed graph optimization problem with guaranteed-optimal classical algorithms. Training a model to approximate what A* already solves exactly would be a strictly worse choice, and is a very good "we know when *not* to use ML" line for judges |
| **Evaluation** | Path cost vs. brute-force optimum on small test graphs; runtime vs. graph size |

---

## 7. Fish Productivity Analysis Agent

| Field | Detail |
|---|---|
| **Question it answers** | "Why has fish productivity changed in this region?" |
| **Input variables** | 30-day rolling window of SST, chlorophyll, current, MLD; fishing effort (vessel-hours); historical catch/landing data by species and location |
| **Dataset source** | Self-constructed: environmental time series (same sources as Ocean Analytics) joined against fisheries landing data (state fisheries departments / CMFRI landing surveys where available) |
| **Target / output** | `productivity(t+1)` forecast, plus a feature-attribution ranking of what's associated with the change |
| **Proposed model** | LSTM/GRU for the forecast; SHAP or permutation importance for attribution |
| **Why this model** | The question is explicitly temporal ("has *changed*"), so a sequence model is the natural fit; attribution methods let you answer the "why" without overclaiming causality |
| **Evaluation** | Forecast RMSE/MAE against held-out months; qualitative check that attributed features match known seasonal patterns (a sanity check, not a formal metric) |
| **Wording caution** | State findings as *"factors most strongly associated with"*, never *"caused"* — this is correlational attribution, not a causal study, and a judge in marine science may probe exactly this distinction |

---

## 8. Planner / Reporting Agent

| Field | Detail |
|---|---|
| **Question it answers** | Everything upstream of the specialist agents: intent parsing, agent routing, and everything downstream: turning structured model outputs into a spoken/written answer |
| **Input variables** | Raw user query (text) for parsing; structured JSON outputs from whichever specialist agents ran, for reporting |
| **Dataset source** | **Marine Query Intent Dataset** (your own — query → intent → required agents/parameters) for routing; a small **Marine QA/Reasoning Dataset** of (structured evidence → natural-language explanation) pairs for reporting |
| **Target / output** | Parsing: `{intent, location, time_range, required_agents}`. Reporting: a natural-language explanation grounded in the upstream scores |
| **Proposed model** | A small general-purpose or lightly fine-tuned language model — not trained from scratch. This is the one place in ORCA where using an existing LLM is the *correct* engineering choice, not a shortcut |
| **Why this model** | Intent parsing and text generation are exactly what LLMs are good at; the domain intelligence has already been produced by the specialist models before the LLM ever sees the query, so the LLM has nothing consequential left to "get wrong" scientifically |
| **Evaluation** | Intent-classification accuracy on your query dataset; for reporting, a rubric check that generated explanations only reference values actually present in the upstream JSON (no hallucinated numbers) |

---

# 9. Model Cards

```text
MODEL: PFZ Suitability Predictor
Arch: CNN-LSTM (fallback: XGBoost)
In:   SST, CHL, currents, MLD (7-day)
Out:  PFZ_score [0,1] per grid cell
Data: INCOIS PFZ + MOSDAC OCM/SST
Eval: F1, spatial IoU


MODEL: Marine Risk Score
Arch: XGBoost / MLP
In:   weather_risk, ocean_score,
      wave, current, lightning,
      cyclone_dist, geofence
Out:  risk_score [0,100] + class
Data: joined Weather+Ocean+Geo
Eval: F1 on DANGER class


MODEL: Ocean Suitability Score
Arch: XGBoost regression
In:   SST, CHL, current, MLD, D20, wave
Out:  suitability [0,1]
Data: INCOIS ROMS + MOSDAC
Eval: RMSE, SHAP feature importance


MODEL: Weather Risk Classifier
Arch: MLP / XGBoost
In:   wind, rain, pressure, humidity,
      visibility, lightning, cyclone
Out:  {NORMAL, CAUTION, DANGEROUS}
Data: ERA5 + IMD API
Eval: F1 per class, calibration


AGENT: Geospatial Reasoning
No ML — PostGIS/GeoPandas
In:   coordinates, GIS layers
Out:  nearest/intersect/contain
Eval: correctness + query latency


AGENT: Route Optimization
No ML — A*/Dijkstra over cost graph
In:   start/end, agent-derived edge
      costs (weather/risk/hazard)
Out:  optimal waypoint list
Eval: path cost vs. brute-force optimum


MODEL: Fish Productivity Forecaster
Arch: LSTM/GRU + SHAP attribution
In:   30-day SST/CHL/current/MLD,
      fishing effort, catch history
Out:  productivity(t+1) + attribution
Data: env. time series + landings
Eval: RMSE, qualitative attribution
      sanity check


MODEL: Planner / Reporting LLM
Arch: small general-purpose LLM
In:   raw query (parse) / upstream
      JSON scores (report)
Out:  {intent, agents} / NL explanation
Data: Query-Intent + QA-Reasoning sets
Eval: intent accuracy, no-hallucination
      rubric on generated explanations
```

---

# 10. The Line to Use With Judges

> "Domain intelligence comes from task-specific models trained on marine, meteorological, and geospatial data — not from a general LLM. Wherever a problem has an exact, deterministic answer, like nearest-neighbour search or shortest-path routing, we use classical algorithms instead of ML, because that's the more reliable choice. The Risk model consumes the *outputs* of the Weather and Ocean models rather than raw sensor data, so risk scoring stays auditable. The language layer sits at both ends: parsing intent at the front, and explaining already-computed scores at the back — it never generates a scientific answer itself."

This sentence pre-empts three separate judge questions at once:

- Why not use an LLM for everything?
- Why train specialized models instead of using a pretrained model?
- Why use classical algorithms for routing/geospatial reasoning?

The third one is a strength: it demonstrates deliberate algorithm selection rather than forcing ML onto every problem.

---

# 11. Full Agent Inventory at a Glance

| # | Agent | Needs ML? | Model Type |
|---|---|---|---|
| 1 | Marine Data Discovery | Yes (light) | Intent classifier |
| 2 | Weather Intelligence | Yes | MLP / XGBoost |
| 3 | Ocean Analytics | Yes | XGBoost regression |
| 4 | PFZ Intelligence | Yes | CNN-LSTM (fallback: XGBoost) |
| 5 | Marine Risk Assessment | Yes | XGBoost / MLP on upstream outputs |
| 6 | Geospatial Reasoning | **No** | PostGIS / GeoPandas |
| 7 | Route Optimization | **No** | A* / Dijkstra |
| 8 | Fish Productivity Analysis | Yes | LSTM/GRU + SHAP |
| 9 | Planner / Reporting | Yes (existing LLM, not from scratch) | Small LLM |

Five agents run trained models, two run deterministic algorithms with no ML at all, and one uses an existing language model where that's genuinely the right tool.

That mix is the actual technical maturity story — not "we trained 8 models," but "we knew which of the 9 problems needed training a model at all."

---

# 12. Suggested Build Order

1. **Weather Intelligence** — cleanest labels (ERA5 + IMD), fastest to prototype, easiest to explain
2. **Ocean Analytics** — feeds PFZ and Risk, so building it second unblocks the other two
3. **Geospatial Reasoning** — no training needed, so it's nearly free to stand up and immediately useful for demos
4. **Marine Risk Assessment** — straightforward once Weather and Ocean exist, since it's mostly feature aggregation
5. **Route Optimization** — also no training needed, and becomes demo-able the moment Risk and Geospatial exist
6. **PFZ Intelligence** — most impressive but most data-engineering-heavy (raster time series); tackle once the pipeline is proven on simpler agents
7. **Fish Productivity Analysis** — the hardest to source real fisheries data for, so time-box this one and be ready to present it as "planned, dataset sourcing in progress" if time runs out
8. **Planner / Reporting** — build last, since it needs the other agents' output shapes finalized before it can consume them
