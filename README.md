# xG Profiler — Shot-Type Specific Finishing Skill

**Beyond Chance Quality: Do Players Have Shot-Specific Finishing Skill?**

BITS Pilani, Goa Campus — Machine Learning Course Project

---

## What This Project Does

Expected Goals (xG) treats every player identically — Messi and a random defender get the same xG for the same shot. This project asks: *do players systematically over or underperform xG in specific shot situations?*

We cluster 490,000 shots from the top 5 European leagues (2014–2025) into 14 situational types using K-Means, then measure each player's finishing residual (Goals − xG) per cluster. The result is a player × cluster heatmap of shot-type-specific finishing skill.

**Data source:** Understat (all xG values are Understat's pre-computed model)
**Leagues:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1
**Seasons:** 2014/15 – 2024/25

---

## The 14 Shot Clusters (V3, K=14, Silhouette=0.533)

| Cluster | Shots | Goal Rate | Mean xG |
|---|---|---|---|
| through_ball_one_on_one | 10,027 | 28.1% | 0.294 |
| rebound_shot | 19,859 | 23.1% | 0.243 |
| inside_box_central | 99,826 | 17.6% | 0.184 |
| top_of_box_edge | 17,150 | 14.1% | 0.140 |
| header_from_cross_corner | 61,392 | 13.6% | 0.153 |
| box_scramble_header | 20,213 | 10.8% | 0.130 |
| set_piece_header | 27,391 | 5.9% | 0.063 |
| direct_free_kick | 17,369 | 6.0% | 0.062 |
| wide_left_box | 26,611 | 4.3% | 0.056 |
| wide_right_box | 29,186 | 4.3% | 0.056 |
| zone14_central_strike | 111,316 | 3.6% | 0.030 |
| long_range_central | 23,334 | 2.2% | 0.020 |
| long_range_left | 22,586 | 2.5% | 0.025 |
| edge_of_box_speculative | 3,944 | 5.7% | 0.053 |

---

## Key Results

### Finishing residual = Actual Goals − Understat xG (per shot, per cluster)

**Standout overperformers:**
- **Federico Chiesa** — `through_ball_one_on_one` (+0.276/shot) — elite in behind-the-defense runs
- **Rodri** — `rebound_shot` (+0.334/shot) — surprising composure on second balls for a midfielder
- **Phil Foden** — `rebound_shot` (+0.332/shot)
- **Serge Gnabry** — `top_of_box_edge` (+0.256/shot) and `rebound_shot` (+0.247/shot)
- **Roberto Firmino** — `box_scramble_header` (+0.181/shot) — fits his movement style exactly
- **James Rodríguez** — `direct_free_kick` (+0.116/shot) — known dead-ball specialist, validates the cluster
- **Isco** — `wide_right_box` (+0.191/shot) — cutting in from the right

**Interesting dual profiles:**
- **Filip Djordjevic** — bottom 3 in `inside_box_central` (-0.226) but top 3 in `header_from_cross_corner` (+0.194) — textbook shot-type specialist
- **Paulo Dybala** — top 3 in `wide_left_box` (+0.134) but bottom 3 in `through_ball_one_on_one` (-0.261)

Full results in `v3_final_results/`.

---

## Repository Structure

```
xg-profiler/
├── src/                        # Pipeline source code
│   ├── scraper.py              # Understat scraper
│   ├── features.py             # Feature engineering
│   ├── clustering.py           # K-Means pipeline
│   ├── residuals.py            # Residual computation
│   └── visualise.py            # Heatmap generation
├── notebooks/                  # Analysis notebooks (run in order)
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_xg_model.ipynb
│   └── 06_residuals_profiles.ipynb
├── v3_final_results/           # All V3 outputs
│   ├── v3_heatmap_top40.png    # Top 40 players by shot volume
│   ├── v3_heatmap_elite.png    # Top 25 elite finishers
│   ├── v3_residuals.csv        # 8,403 player-cluster residual pairs
│   ├── v3_cluster_summary.csv  # 14 clusters with goal rates + mean xG
│   ├── kmeans_v3_model.pkl     # Fitted K-Means model
│   └── elbow_silhouette.png    # K selection chart
├── empirical_verification/     # Manual footage verification (see below)
│   ├── through_ball_one_on_one/
│   ├── rebound_shot/
│   ├── inside_box_central/
│   └── ... (14 cluster folders)
├── CLAUDE.md                   # Full project context and methodology
├── requirements.txt
└── README.md
```

---

## Empirical Verification — Task for the Group

The `empirical_verification/` folder contains CSV files for manual spot-checking that the clustering is working correctly.

**What to do:**
For each cluster folder, open the `TOP_` and `BOTTOM_` CSV files. Each row is one shot with:
- `date` — match date
- `h_team` / `a_team` — home and away team
- `minute` — when the shot was taken
- `xg_understat` — Understat's xG for that shot
- `goal_label` — GOAL or NO_GOAL
- `result` — SavedShot, MissedShots, BlockedShot, Goal

**The check:** Find the match on YouTube/Sofascore/footchamps, go to that minute, and confirm the shot situation matches the cluster label. For example, a shot in `through_ball_one_on_one` should clearly show a through ball with the player in a 1v1 against the keeper.

**Files:** 6 files per cluster (TOP_Player1, TOP_Player2, TOP_Player3, BOTTOM_Player1, BOTTOM_Player2, BOTTOM_Player3), 14 clusters = 84 files total.

**Priority clusters to check** (most interesting for the paper):
1. `through_ball_one_on_one` — does it actually capture 1v1 situations?
2. `rebound_shot` — are these genuine second-ball situations?
3. `direct_free_kick` — easiest to verify visually
4. `box_scramble_header` — headers not from crosses, e.g. flick-ons

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Run notebooks in order (01 → 06). Raw data not included in the repo due to size — run `01_data_collection.ipynb` to scrape from Understat.

---

## Methodology

**Stage 1 — Clustering (unsupervised):** K-Means on 19 features (distance, angle, 6 spatial zone flags, 12 situational binary flags). K=14 chosen by elbow method + silhouette score (0.533). Foot features excluded — they caused trivial body-part clusters in earlier runs.

**Stage 2 — Residuals (supervised baseline):** For each player with ≥15 shots in a cluster, compute `residual = actual_goals − sum(xG)`. xG values are Understat's pre-computed model. Minimum 15 shots per player-cluster cell to report (10 for the small `edge_of_box_speculative` cluster).

**Robustness:** Findings are measured against Understat xG, the industry standard model for the leagues covered.

---

## Related Work

- Davis & Robberechts (2024) — "Biases in Expected Goals Models Confound Finishing Ability" — arXiv:2401.09940
- Scholtes & Karakuş (2024) — "Bayes-xG: Player and Position Correction on Expected Goals Using Bayesian Hierarchical Approach"
- StatsBomb blog — "Quantifying Finishing Skill"
