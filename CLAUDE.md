# CLAUDE.md — Shot-Type Specialist Finishing Profiles

This file gives you full context on this ML project. Read it before doing anything else.

---

## Project summary

**Title:** Beyond Chance Quality: Do Players Have Shot-Specific Finishing Skill?

**One-line pitch:** We cluster football shots into situational types using unsupervised ML, then use a supervised xG model to measure whether players systematically over or underperform within specific shot types — decomposing "good finisher" into something more granular than any published model.

**Course:** Machine Learning (BITS Pilani, Goa Campus)
**Deadline:** Final evaluation last week of April 2025
**Team size:** 7 members

---

## The research question

Expected Goals (xG) is a probability score for each shot in football — it estimates how likely an average player is to score from that situation, based on features like distance, angle, body part, defensive pressure, etc. It's modelled as logistic regression.

The problem: xG treats every player the same. Messi and a random defender get the same xG for the same shot. Some papers have tried adding a single per-player finishing coefficient, but that flattens everything into one number. A player might be elite at tap-ins but average from long range.

**Our question: do players have shot-type-specific finishing ability, and can we measure it?**

This specific framing — shot clustering + per-cluster residuals — has not been done in published literature. The closest work uses Bayesian hierarchical logistic regression with a single player coefficient, or position-adjusted xG submodels. Neither decomposes at the shot-situation level.

---

## Clustering versions

### V2 — CURRENT PRODUCTION (zone-based spatial features)

**K=12, silhouette=0.4960**
**19 features:** 13 situational flags + 6 spatial zone binary flags

Feature set:
```
distance, angle, is_header,
from_corner, from_set_piece, from_freekick,
preceded_by_cross, preceded_by_aerial, preceded_by_dribble,
preceded_by_throughball, preceded_by_rebound, preceded_by_layoff,
shot_in_box,
zone_13, zone_14, zone_15, zone_16, zone_17, zone_18
```

**Key methodological decision — why V2 over V1:** V1 included `is_right_foot` and `is_left_foot`. These are strong orthogonal binary features, so K-Means latched onto them as the primary separator and produced trivial body-part clusters (right foot shots vs left foot shots) rather than situational clusters. Removing them and replacing with spatial zone flags forces the algorithm to cluster by *where on the pitch and in what situation*, which is the actual signal of interest.

**Zone setup:** 18-zone grid (6 columns × 3 rows) on the 0–100 Understat pitch. Zones 13–18 are the final third (closest to goal), covering 90%+ of all shots. Zone formula: `zone_id = floor(x*6/100)*3 + floor(y*3/100) + 1`. Zones 13–18 are row 4 and 5 of the grid (x > 66.7).

**V2 clusters (K=12):**

| cluster_id | cluster_name | n | goal% | Description |
|---|---|---|---|---|
| 0 | long_range_central | 23,337 | 2.2% | Long range, central (zone 15) |
| 1 | zone14_central_strike | 119,559 | 3.7% | Straight-on shot just outside box (zone 14) |
| 2 | inside_box_central | 115,406 | 17.7% | Standard inside-box chance (zone 17) |
| 3 | long_range_left | 22,587 | 2.5% | Long range, left flank (zone 13) |
| 4 | medium_range_zone14 | 3,944 | 5.7% | Small zone 14 medium-range bucket |
| 5 | wide_left_box | 26,629 | 4.3% | Tight angle, left side of box (zone 16) |
| 6 | set_piece_header | 32,157 | 5.7% | Corner/set piece aerial header |
| 7 | wide_right_box | 28,691 | 4.3% | Tight angle, right side of box (zone 18) |
| 8 | header_from_cross | 70,012 | 13.6% | Cross delivered, headed finish (zone 17) |
| 9 | through_ball_one_on_one | 10,027 | 28.1% | Through ball, 1v1 situation — highest goal rate |
| 10 | medium_range_zone14_b | 17,369 | 6.0% | Mid-distance zone 14 variant |
| 11 | rebound_shot | 20,486 | 22.5% | Shot preceded by rebound — second highest goal rate |

**V2 outputs:**
- `data/outputs/kmeans_v2_model.pkl` — fitted KMeans + scaler + feature list
- `data/outputs/v2_cluster_labels.csv` — 490k shots with v2_cluster_id and v2_cluster_name
- `data/outputs/v2_residuals.csv` — 8,433 player-cluster pairs (≥15 shots threshold)
- `data/outputs/v2_heatmap_top40.png` — top 40 players by volume
- `data/outputs/v2_heatmap_elite.png` — top 25 players by avg residual
- `data/processed/shots_features_v2.csv` — full feature matrix with V2 labels attached

**Standout V2 findings:**
- **Federico Chiesa** tops through_ball_one_on_one (+0.33 residual/shot) — elite in behind-the-defense runs
- **Rodri** tops rebound_shot (+0.34/shot) — surprising for a midfielder, shows composure on second balls
- **Erling Haaland** appears in zone14_central_strike (+0.15/shot) — clinical from central outside-box range, not just tap-ins
- **Achraf Hakimi** and **Isco** surface in wide box clusters — overlapping fullbacks and inside forwards cutting in
- **Gareth Bale** top 5 in through_ball_one_on_one — consistent with counter-attacking profile

---

### V3 — CURRENT PRODUCTION (K=14, zone + situational, best silhouette)

**K=14, silhouette=0.5328**
**19 features:** same as V2 (zone flags replaced foot features — no change from V2 feature set)

V3 is V2 with K increased from 12 to 14. The extra two K values produced three genuinely new clusters that V2 couldn't isolate:

| cluster_id | cluster_name | n | goal% | What's new vs V2 |
|---|---|---|---|---|
| 0 | edge_of_box_speculative | 3,944 | 5.7% | Small ambiguous bucket |
| 1 | inside_box_central | 99,826 | 17.6% | Same as V2 |
| 2 | zone14_central_strike | 111,316 | 3.6% | Same as V2 |
| 3 | header_from_cross_corner | 61,392 | 13.6% | Cleaner than V2 (cross+corner combined) |
| 4 | wide_left_box | 26,611 | 4.3% | Same as V2 |
| 5 | long_range_central | 23,334 | 2.2% | Same as V2 |
| 6 | wide_right_box | 29,186 | 4.3% | Same as V2 |
| 7 | set_piece_header | 27,391 | 5.9% | Same as V2 |
| 8 | long_range_left | 22,586 | 2.5% | Same as V2 |
| 9 | direct_free_kick | 17,369 | 6.0% | **NEW** — freekick=1.0, zone_14=0.70 |
| 10 | through_ball_one_on_one | 10,027 | 28.1% | Same as V2 |
| 11 | top_of_box_edge | 17,150 | 14.1% | **NEW** — distance=19.8, straddles zone14/zone17 (the "D") |
| 12 | box_scramble_header | 20,213 | 10.8% | **NEW** — headers not from cross (flick-ons, second balls) |
| 13 | rebound_shot | 19,859 | 23.1% | Same as V2 |

**V3 outputs:**
- `data/outputs/kmeans_v3_model.pkl`
- `data/outputs/v3_cluster_labels.csv`
- `data/outputs/v3_residuals.csv` — 8,405 player-cluster pairs (≥15 shots)
- `data/outputs/v3_heatmap_top40.png`
- `data/outputs/v3_heatmap_elite.png`
- `data/processed/shots_features_v3.csv`

**Standout V3 findings:**
- **Federico Chiesa** tops through_ball_one_on_one (+0.33/shot)
- **Rodri** tops rebound_shot (+0.34/shot)
- **Serge Gnabry** tops top_of_box_edge (+0.27/shot)
- **Roberto Firmino** tops box_scramble_header (+0.24/shot) — fits his movement style
- **Griezmann** and **James Rodríguez** top direct_free_kick — known dead-ball specialists, validates the cluster
- **Erling Haaland** in zone14_central_strike top 5 (+0.14/shot)

**direct_free_kick is the most report-worthy new finding** — shot quality is nearly fixed by wall/distance, yet players still show significant residuals, which is strong evidence for shot-type-specific skill.

---

### V2 — Kept for reference (K=12, silhouette=0.4960)

**K=6, silhouette=lower, 15 features** (includes is_right_foot and is_left_foot)

V1 produced clusters dominated by body part rather than shot situation — K-Means split primarily into right-foot and left-foot groups, which is technically valid but analytically useless. Kept for reference and comparison in the report to motivate the V2 redesign.

V1 outputs: `data/outputs/cluster_labels.csv`, `data/outputs/residuals.csv`, `data/outputs/heatmap_top40.png`

---

## ML pipeline (two stages)

### Stage 1 — Shot clustering (unsupervised ML)

**Algorithm:** K-Means
**Goal:** Group shots into situational types without any predefined labels
**K selection:** Run K=3 to 10, use elbow method + silhouette score, pick optimal K (expect 5–7 clusters)
**Cluster naming:** After fitting, inspect centroids and name clusters manually based on dominant features

Expected clusters (these are hypotheses, not fixed labels — let the data decide):
- Close-range first-time shot (central, low angle, under pressure)
- Header from cross (aerial, wide origin, set play or open play)
- Long-range strike (distance > 25 yards, right or left foot)
- Cutback tap-in (low angle, first touch, preceded by cut-back pass)
- Set piece header (corner or free kick origin, header)
- One-on-one with goalkeeper (one_on_one flag, close range)

**Important:** These cluster labels do not exist in the raw data. We derive them. That is the unsupervised contribution of this project.

### Stage 2 — xG model + residual analysis (supervised ML)

**Models:**
- Logistic Regression (baseline — simple, interpretable, standard in xG literature)
- XGBoost (main model — handles non-linear feature interactions)
- Both evaluated with 5-fold cross-validation

**Evaluation metrics for xG model:**
- Brier score (calibration — most important for probability models)
- ROC-AUC (discrimination)
- Calibration plots (visual check that predicted probabilities match observed rates)
- Log loss

**Second xG baseline:** Understat provides pre-computed xG values for every shot. Use these alongside our model as a robustness/sensitivity check. If findings hold under both xG baselines, they are robust to model choice.

**Residual computation:**
```
Residual = Actual Goals − Sum(xG)   [per player, per cluster]
```

Positive residual = player scores more than expected in that cluster = finishing specialist there.
Negative residual = player underperforms in that cluster.

**Sample size threshold:** Minimum 15 shots per player per cluster to report a residual. Below this, do not report — insufficient sample. This threshold should be mentioned explicitly in the report as a methodological decision.

**SHAP values:** Compute SHAP feature importance for the XGBoost model. This shows which shot features drive xG predictions most — important for the evaluation and interpretation section.

**Output:** A player × cluster heatmap where colour intensity = finishing residual. This is the core deliverable.

---

## Data sources

### Primary — Understat

- **URL:** understat.com
- **Coverage:** All top 5 European leagues — Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- **Seasons:** 2014/15 through 2024/25 (confirmed by domain expert)
- **Granularity:** Shot-level — each row is one shot
- **Key fields available:** x, y coordinates, result (goal/no goal), situation (open play, corner, free kick, set piece), shot type (foot, header), player, match, date, xG (Understat's own model)
- **Access:** Free, no account needed, scrapeable via Python
- **Scrapers:** Multiple well-documented Python scrapers on GitHub — search `understatAPI` or `understat python scraper`. The `understat` PyPI package is the cleanest option.
- **Volume:** Expect 100,000+ shots across all leagues and seasons

**How to scrape Understat:**
```python
pip install understat
```
```python
import asyncio
import understat

async def main():
    async with understat.Understat() as u:
        # Get all shots for a league/season
        data = await u.get_league_players("EPL", 2023)
        shots = await u.get_player_shots(player_id)
asyncio.run(main())
```

Alternatively scrape directly from the JSON embedded in match pages. Multiple approaches work — use whichever the scraper you find handles cleanest.

### Secondary — StatsBomb Open Data

- **URL:** github.com/statsbomb/open-data
- **Install:** `pip install statsbombpy`
- **Coverage:** La Liga (15+ seasons, heavily Messi-era Barca), Champions League, Women's World Cup, EURO 2020, FA Cup, NWSL
- **Key advantage over Understat:** Freeze frame data — for every shot, StatsBomb records the position of every player on the pitch at the moment of the shot. This lets you compute:
  - Number of defenders in the shot triangle (between shooter and goalposts)
  - Goalkeeper distance to goal
  - Proportion of goal visible (unobstructed)
  - Whether GK was out of position
- **Use case in this project:** Supplement Understat data with richer contextual features where available. Also use StatsBomb's pre-attached `shot_statsbomb_xg` as an additional xG baseline.
- **Access:** Free, no credentials needed for open data

**Key shot fields in StatsBomb:**
- `shot_body_part` — Right Foot, Left Foot, Head
- `shot_technique` — Normal, Volley, Half Volley, Lob, Backheel
- `shot_type` — Open Play, Corner, Free Kick, Penalty
- `shot_first_time` — Boolean
- `shot_one_on_one` — Boolean
- `shot_outcome` — Goal, Saved, Missed, Blocked, Off T
- `under_pressure` — Boolean
- `shot_statsbomb_xg` — StatsBomb's pre-computed xG value
- `shot_freeze_frame` — Nested array of player positions at shot moment
- `play_pattern_name` — From Counter, Regular Play, From Corner, etc.
- `location` — [x, y] coordinates (pitch is 120×80 units, origin bottom-left)

### Nutmeg (Claude Code plugin — tooling, not data)

- **Install inside Claude Code:** `/plugin install nutmeg@withqwerty/nutmeg`
- **What it does:** Gives Claude Code verified, up-to-date documentation on StatsBomb, Understat, Opta, and other providers so it doesn't hallucinate field names, coordinate systems, or API signatures
- **Useful skills:**
  - `/nutmeg-acquire` — helps write and debug the Understat scraper and StatsBomb parser
  - `/nutmeg-wrangle` — helps with coordinate normalisation, flattening nested JSON freeze frames, joining datasets
  - `/nutmeg-heal` — fixes broken scrapers
  - `/nutmeg-compute` — derived metrics like xG, PPDA
- **Caveat:** New and lightly tested (1 star, 23 commits as of project start). Use as a productivity accelerator, not a dependency. If it gives wrong output, fall back to manual.

---

## Feature engineering

These are the features to engineer per shot. Some come directly from Understat, some require computation, some require StatsBomb freeze frames.

| Feature | Source | Type | Notes |
|---|---|---|---|
| Distance to goal | Understat x/y or StatsBomb location | Continuous | Euclidean from shot location to goal centre |
| Shot angle | Understat x/y or StatsBomb location | Continuous | Angle between shooter and goalposts |
| Body part | StatsBomb `shot_body_part` / Understat `shot_type` | Categorical | Encode: Right Foot, Left Foot, Head |
| Shot technique | StatsBomb `shot_technique` | Categorical | Normal, Volley, Half Volley, Lob, Backheel |
| First touch | StatsBomb `shot_first_time` | Binary | True/False |
| Under pressure | StatsBomb `under_pressure` / Understat situation | Binary | True/False |
| One on one | StatsBomb `shot_one_on_one` | Binary | True/False |
| Play pattern | StatsBomb `play_pattern_name` | Categorical | From Counter, Regular Play, From Corner, etc. |
| Shot type | StatsBomb `shot_type` / Understat situation | Categorical | Open Play, Corner, Free Kick, Penalty |
| Follows dribble | StatsBomb `shot_follows_dribble` | Binary | True/False |
| GK distance to goal | StatsBomb freeze frame | Continuous | Compute from GK coordinates |
| Defenders in triangle | StatsBomb freeze frame | Integer | Count defenders between shooter and posts |
| Free goal proportion | StatsBomb freeze frame | Continuous | Proportion of goal unobstructed |
| Game state | Match score at time of shot | Categorical | Winning/Drawing/Losing — compute from match events |

**Understat coordinate system:** x=0 to 100, y=0 to 100 (percentage of pitch length/width)
**StatsBomb coordinate system:** x=0 to 120, y=0 to 80 (yards). Origin is bottom-left corner.

**Normalise coordinates** before clustering — scale to consistent units. Use StandardScaler on all continuous features before running K-Means.

---

## Project structure (suggested)

```
project/
├── CLAUDE.md                    ← this file
├── data/
│   ├── raw/
│   │   ├── understat/           ← raw scraped JSON/CSV from Understat
│   │   └── statsbomb/           ← StatsBomb open data (downloaded or via API)
│   ├── processed/
│   │   ├── shots_clean.csv      ← merged, cleaned shot-level dataset
│   │   └── shots_features.csv   ← feature-engineered dataset ready for ML
│   └── outputs/
│       ├── cluster_labels.csv      ← V1 cluster labels (reference)
│       ├── residuals.csv           ← V1 residuals (reference)
│       ├── v2_cluster_labels.csv   ← V2 cluster labels (CURRENT)
│       ├── v2_residuals.csv        ← V2 residuals (CURRENT)
│       ├── xg_predictions.csv      ← model xG per shot
│       ├── kmeans_v2_model.pkl     ← fitted V2 KMeans + scaler
│       └── v2_heatmap_elite.png    ← main visual output
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_xg_model.ipynb
│   └── 06_residuals_profiles.ipynb
├── src/
│   ├── scraper.py               ← Understat scraper
│   ├── features.py              ← feature engineering functions
│   ├── clustering.py            ← K-Means pipeline
│   ├── xg_model.py              ← logistic regression + XGBoost
│   ├── residuals.py             ← per-player per-cluster residual computation
│   └── visualise.py             ← heatmap and plots
├── requirements.txt
└── README.md
```

---

## Key decisions already made

These were deliberated and settled — do not relitigate unless there's a strong technical reason:

1. **Primary data source is Understat**, not StatsBomb, because it covers all top 5 leagues since 2014/15. StatsBomb open data is supplementary for freeze-frame features.

2. **Do not build an xG model from scratch as the sole baseline.** Use Understat's pre-computed xG as one baseline. Build a logistic regression as a second baseline. Use both for robustness checking. This is framed as a sensitivity analysis, not a competition.

3. **Our logistic regression xG is not competing with Understat's model.** It is a simplified stress-test. If our findings hold under both, they are model-agnostic.

4. **Penalties are excluded** from all analysis. They are a fixed situation (xG ≈ 0.76) and would distort the clustering. Filter them out early in the pipeline.

5. **Minimum 15 shots per player per cluster** to report a residual. Below this threshold, mark as insufficient sample. Do not impute or estimate.

6. **Cluster labels are assigned post-hoc by human inspection**, not by the algorithm. K-Means gives cluster IDs (0, 1, 2...) — we look at the centroids and name them meaningfully. Document this process.

7. **No deep learning.** Hard constraint from the course. All models must be classical ML — logistic regression, XGBoost, K-Means, and their variants.

8. **No tracking data.** We are not using Opta or any paid tracking provider. All data is free and publicly accessible.

---

## Evaluation framework

### Clustering
- Elbow method plot (inertia vs K)
- Silhouette score for chosen K
- Cluster centroid inspection — describe what each cluster represents
- Qualitative validation — do the cluster names match football intuition?

### xG model
- Brier score (primary — calibration matters most for probability outputs)
- ROC-AUC
- 5-fold cross-validation
- Calibration plot (predicted probability vs observed goal rate)
- SHAP feature importance plot
- Compare our model's predictions vs Understat xG on held-out shots

### Finishing profiles
- Per-player per-cluster residual table (filtered to ≥15 shots)
- Robustness check: do top/bottom finishers per cluster agree across both xG baselines?
- Finishing profile heatmap (main visual output)
- Case study: pick 3–5 known elite finishers and walk through their profile

### Limitations to acknowledge
- Sample size per cluster is uneven — some players don't have enough shots in certain clusters
- StatsBomb open data covers limited leagues; freeze frame features only available for those competitions
- Understat xG is itself a model with its own biases — we are computing residuals against an imperfect baseline
- Regression to the mean — players with few shots will have noisy residuals

---

## Libraries and environment

```
python >= 3.9
pandas
numpy
scikit-learn          # K-Means, logistic regression, StandardScaler, cross-validation
xgboost               # XGBoost model
shap                  # SHAP feature importance
matplotlib
seaborn               # Heatmap visualisation
statsbombpy           # StatsBomb open data
understat             # Understat async scraper
tqdm                  # Progress bars for scraping loops
jupyter
```

Install all:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn statsbombpy understat tqdm jupyter
```

---

## Domain knowledge for Claude Code

These are football-specific facts that are easy to get wrong:

- **A goal in football = 1 point.** Shots either result in a goal or they don't. xG is binary classification.
- **Pitch coordinates:** StatsBomb uses 120×80 yards. Goal is at x=120, y=36 to y=44. Attacking direction is left to right (increasing x). Understat uses 0–100 percentage scale for both axes.
- **xG range:** Always between 0 and 1. A penalty is approximately 0.76 xG. A tap-in from 2 yards is ~0.8–0.9. A long-range shot is ~0.02–0.05.
- **GAX (Goals Above Expected):** Our residual is essentially GAX but computed per cluster, not in aggregate. This is the key novel contribution.
- **Shot situation in Understat:** The `situation` field takes values like `OpenPlay`, `FromCorner`, `SetPiece`, `DirectFreekick`, `Penalty`.
- **Shot result in Understat:** The `result` field takes values like `Goal`, `SavedShot`, `MissedShots`, `BlockedShot`, `OwnGoal`.
- **Headers:** A significant portion of goals come from headers, especially from set pieces. They behave very differently from foot shots in xG models — always encode body part separately.
- **Freeze frames in StatsBomb:** The `shot_freeze_frame` field is a list of dicts, each with `location` [x, y], `teammate` (bool), `actor` (bool), `keeper` (bool), `position` (dict with name). Parse this carefully — it's nested JSON inside a DataFrame column.

---

## What good output looks like

At the end of this project you should be able to answer questions like:

- "Which active forwards are the most clinical finishers from close range first-time shots?"
- "Which players consistently underperform their xG on headers?"
- "Is there a player who is average overall but elite in one specific shot type?"
- "Do the findings change if we use Understat xG vs our own model?"

The heatmap should make these questions answerable visually in under 5 seconds.

---

## Related work (for context and citations)

- **Davis & Robberechts (2024)** — "Biases in Expected Goals Models Confound Finishing Ability" — arXiv:2401.09940. Key finding: standard GAX is biased by shot volume and position. Our work addresses this by conditioning on shot type cluster.
- **Scholtes & Karakuş (2024)** — "Bayes-xG: Player and Position Correction on Expected Goals Using Bayesian Hierarchical Approach" — Frontiers in Sports and Active Living. Uses Bayesian hierarchical model with player effects. We use a frequentist residual approach conditioned on clusters instead.
- **Bandara et al. (2024)** — "Predicting Goal Probabilities with Improved xG Models Using Event Sequences" — PLoS One. Adds sequence-based features to xG. We focus on shot-type specialisation rather than sequence modelling.
- **StatsBomb blog** — "Quantifying Finishing Skill" — introduces the concept of finishing skill as a player-level effect. Our work extends this to shot-type-level.

---

## Notes on tone and framing

- This is a **course project** for a Machine Learning course, audience is the course professor
- The professor may not be familiar with football analytics terminology — explain xG and related concepts where they appear in outputs or the report
- Frame contributions as: novel research question + two ML paradigms combined + interpretable real-world output
- The robustness check (two xG baselines) is a deliberate methodological choice, not a workaround — frame it as such
- The project title for submission: **"Beyond Chance Quality: Do Players Have Shot-Specific Finishing Skill?"**
