# Analysis Reference

This document maps the final **research questions, code, inputs, methods, outputs, report figures, demo views, and interpretation boundaries**.

It is intended as the quickest technical bridge between the final report and the repository.

---

# Project map

| Problem | Script | Primary input | Main output |
| --- | --- | --- | --- |
| 1. Temporal behavior | `code/02_q1_analysis.py` | Q1 target timelines | `data/analysis/q1/` |
| 2. Champion pairings | `code/03_q2_pairings.py` | canonical participant Parquet | `data/analysis/q2/` |
| 3. Champion network | `code/04_q3_network.py` | Q2 pair/champion/role tables | `data/analysis/q3/` |

Preparation and orchestration:

```text
code/01_prepare_data.py
code/run_project.py
```

Final reports:

```text
docs/report/Report.pdf
docs/report/Report_noimages.pdf
```

Interactive presentation layer:

```text
demo/app.py
```

---

# Problem 1 — Temporal Behavior & Next-Match Performance

## Research question

> How are recent competitive volume, session depth, and post-loss requeue timing associated with performance in a player's subsequent Ranked Solo/Duo match?

## Analysis population

Main target:

```text
queue 420
target duration >=10 minutes
has prior ranked history
target win observed
```

Observed history:

```text
queues 420 + 440
```

Main cohort:

```text
authoritative tracked players
```

Robustness cohort:

```text
alias_confirmed
```

## Main exposures

### H1 — Session depth

Primary session boundary:

```text
30 minutes
```

Sensitivity:

```text
45 / 60 / 90 minutes
```

### H2 — Post-loss requeue timing

Primary categorical bins from:

```text
<=5m
...
>24h
```

with continuous `log2(1 + gap)` sensitivity.

### H3 — Recent ranked volume

Primary window:

```text
6 hours
```

Sensitivity:

```text
3 / 12 / 24 hours
```

## Inference

Primary model:

- linear probability model for target win;
- player fixed effects via within-player demeaning;
- historical and patch controls;
- two-way clustered standard errors by player and physical match;
- Holm correction within predefined behavior-effect families.

Interpretation:

```text
percentage-point change in next-match win probability
```

## Prediction

Three entropy-tree feature sets:

```text
history
behavior
combined
```

Chronological split within region:

```text
70% train / 15% validation / 15% test
```

Validation grid:

```text
max_depth = 2,3,4,5,6,8
min_samples_leaf = 250,1000,3000
```

## Frozen key results

| Metric | Result |
| --- | ---: |
| Primary target observations | **1,146,681** |
| History tree test ROC-AUC | **0.5138** |
| Behavior tree test ROC-AUC | **0.5123** |
| Combined tree test ROC-AUC | **0.5185** |
| Combined − history AUC | **+0.0047** |
| Holm-significant robustness terms | **0** |
| Max >=5m coefficient change | **0.0922 pp** |

Selected trees:

```text
history  -> depth 6, leaf 250
behavior -> depth 2, leaf 3000
combined -> depth 6, leaf 3000
```

## Main conclusion

There is little evidence for a universal short-term fatigue/session-depth/post-loss penalty.

The estimated behavioral effects are small and inconsistent across regions, and behavioral timing adds only limited held-out predictive information beyond recent historical performance.

## Report figures

| Figure | File | Purpose |
| --- | --- | --- |
| 1 | `data/analysis/q1/figures/report/figure_1_inter_match_gap_ecdf.png` | justify session-threshold sensitivity |
| 2 | `data/analysis/q1/figures/report/figure_2_adjusted_behavior_effects.png` | adjusted H1/H2/H3 estimates + 95% CIs |
| 3 | `data/analysis/q1/figures/report/figure_3_prediction_evaluation.png` | held-out ROC + confusion matrix |
| 4 | `data/analysis/q1/figures/report/figure_4_feature_importance.png` | combined-tree interpretation |
| 5 | `data/analysis/q1/figures/report/figure_5_tree_top_levels.png` | readable top levels of entropy tree |

## Demo view

`demo/app.py` uses:

```text
data/analysis/q1/tables/key_results_for_report.csv
data/analysis/q1/tables/behavior_effects.csv
data/analysis/q1/figures/report/figure_3_prediction_evaluation.png
```

The demo emphasizes the **weak behavioral signal** rather than presenting the tree as a strong win predictor.

## Interpretation boundary

Report:

```text
association / prediction
```

Do not claim:

```text
causal fatigue
measured psychological tilt
```

---

# Problem 2 — Champion Pairings & Combo Performance

## Research question

> Which champion pairs are selected together more often than expected, and how is co-selection strength related to pair performance?

## Input

```text
data/processed/full_{na,kr,eu}/participants/*.parquet
```

Filter:

```text
queue 420
duration >=10 minutes
valid five-champion teams
```

## Analytical unit

One unordered champion pair within a valid team.

Each five-champion team contributes:

```text
10 pair observations
```

before aggregation.

## Measures

### Raw co-pick frequency

```text
games_together
```

### Expected co-picks

```text
expected = appearances_A × appearances_B / valid_teams
```

### Lift

```text
lift = observed / expected
```

### Normalized association

```text
association = log2(lift)
```

### Descriptive pair win surplus

```text
100 × [pair win rate − mean(individual champion win rates)]
```

## Support thresholds

```text
>=500 games  -> combo landscape
>=1000 games -> high-support ranking/performance comparisons
```

## Representative results

Most common raw pairs include:

```text
Lucian + Nami
Nautilus + Kaisa
Diana + Yasuo
Kaisa + Sylas
Lulu + Yunara
```

Strong normalized pairings include:

```text
Zeri + Yuumi
Rakan + Xayah
KogMaw + Lulu
Twitch + Yuumi
Lucian + Nami
```

These lists illustrate why raw popularity and normalized association must be treated separately.

## Main conclusion

Raw frequency, normalized co-selection, and pair performance measure different properties.

A pair can be unusually common together after popularity normalization without having unusually high descriptive win surplus.

## Report figures

| Figure | File | Purpose |
| --- | --- | --- |
| 6 | `data/analysis/q2/figures/report/figure_6_pair_rankings.png` | raw popularity vs normalized association |
| 7 | `data/analysis/q2/figures/report/figure_7_combo_landscape.png` | association vs descriptive win surplus |

Supplementary:

```text
data/analysis/q2/figures/supplementary/pair_win_surplus.png
```

## Demo view

The interactive champion selector reads:

```text
data/analysis/q2/tables/champion_stats.csv
data/analysis/q2/tables/pair_stats.csv
```

For one selected champion it shows:

- most common partner;
- strongest normalized partner;
- high-support win-surplus partner;
- association-vs-performance scatter;
- sortable partner table.

## Interpretation boundary

Use:

```text
co-selection
association
descriptive win surplus
```

Avoid:

```text
proven synergy
causal champion interaction
optimal duo recommendation
```

---

# Problem 3 — Champion Network Structure & Team-Composition Communities

## Research question

> What larger structural patterns emerge from the champion co-selection network, and which champions and communities occupy central roles within team compositions?

## Inputs

```text
data/analysis/q2/tables/champion_stats.csv
data/analysis/q2/tables/pair_stats.csv
data/analysis/q2/tables/role_counts.csv
```

Problem 3 intentionally reuses Problem 2's compact outputs.

## Primary graph

Node:

```text
champion
```

Edge retained when:

```text
games_together >=500
association >0
```

Primary edge weight:

```text
log2(lift)
```

## Louvain

Louvain identifies dense groups on the largest supported association component.

The report currently summarizes **7 communities**.

Role composition is compared using:

```text
Top / Jungle / Mid / ADC / Support
```

## Association-weighted PageRank

Representative high-centrality champions include:

```text
Lulu
Milio
Jinx
Caitlyn
Kaisa
Ziggs
Diana
Orianna
Nami
```

The main report uses normalized-association PageRank rather than raw-frequency PageRank.

## Cliques

Clique analysis keeps the strongest association edges and finds fully connected groups of size >=3.

Representative high-association cliques include:

```text
Lulu | XinZhao | Yunara
Jinx | Milio | Volibear
Caitlyn | Lux | Volibear
Jinx | Milio | Mordekaiser
Jinx | Milio | Nasus
Kaisa | LeeSin | Neeko
```

## Comparison methods

Supplementary comparisons include:

```text
Girvan-Newman
K-Means++
Ward hierarchical clustering
DBSCAN
PCA
cosine profile similarity
```

K-Means model selection favored a coarse `k=2` profile partition by silhouette score, while graph methods provided the more interpretable relational view used in the report.

## Main conclusion

The normalized champion network contains interpretable role-related communities, structurally central champions, and recurring higher-order groups.

Graph methods are the preferred primary representation because they preserve the pair relationships directly.

## Report figures

| Figure | File | Purpose |
| --- | --- | --- |
| 8 | `data/analysis/q3/figures/report/figure_8_louvain_community_network.png` | visual community structure |
| 9 | `data/analysis/q3/figures/report/figure_9_community_role_heatmap.png` | role interpretation of communities |
| 10 | `data/analysis/q3/figures/report/figure_10_pagerank_association.png` | central champions |
| 11 | `data/analysis/q3/figures/report/figure_11_strongest_cliques.png` | higher-order co-selection motifs |

## Demo view

The champion-network tab reads:

```text
data/analysis/q3/tables/centrality.csv
data/analysis/q3/tables/louvain_communities.csv
data/analysis/q3/tables/community_role_percent.csv
data/analysis/q3/tables/maximal_cliques.csv
data/analysis/q3/figures/report/figure_8_louvain_community_network.png
```

It also reuses Q2 `pair_stats.csv` to display a selected champion's strongest supported network neighbors.

## Interpretation boundary

Use:

```text
network community
structural centrality
co-selection motif
role complementarity
```

Avoid:

```text
strategic intent
causal synergy
guaranteed optimal composition
```

---

# Final report-to-code map

```text
Problem 1 -> code/02_q1_analysis.py -> Figures 1-5
Problem 2 -> code/03_q2_pairings.py -> Figures 6-7
Problem 3 -> code/04_q3_network.py -> Figures 8-11
```

Final report:

```text
docs/report/Report.pdf
```

Text-only report:

```text
docs/report/Report_noimages.pdf
```

Interactive demo:

```text
demo/app.py
```

---

# Generated artifact map

```text
Problem 1
  data/analysis/q1/tables/
  data/analysis/q1/figures/
  data/analysis/q1/audit/

Problem 2
  data/analysis/q2/tables/
  data/analysis/q2/figures/
  data/analysis/q2/summary.json

Problem 3
  data/analysis/q3/tables/
  data/analysis/q3/figures/
  data/analysis/q3/summary.json
```

Large Q1 timelines and prediction Parquet are intentionally treated as generated intermediates rather than presentation artifacts.

The report, README, technical documentation, and Streamlit dashboard all point back to the same three finalized analyses.
