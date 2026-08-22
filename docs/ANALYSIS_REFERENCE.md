# Analysis Reference

A concise map from the **three report problems** to their code, analytical unit, methods, outputs, and interpretation. This is intended as a quick reference for reviewers who want to connect the report to the implementation.

---

## Overview

| Problem | Scientific unit | Script | Main output folder |
| --- | --- | --- | --- |
| **1. Temporal Behavior & Next-Match Performance** | tracked player × target match | `code/02_q1_analysis.py` | `data/analysis/q1/` |
| **2. Champion Pairings & Combo Performance** | unordered same-team champion pair | `code/03_q2_pairings.py` | `data/analysis/q2/` |
| **3. Champion Network Structure & Communities** | champion graph / community | `code/04_q3_network.py` | `data/analysis/q3/` |

`code/01_prepare_data.py` is a shared prerequisite for Problem 1 only. Problems 2 and 3 deliberately use the broader team-composition corpus rather than the tracked-player cohort.

---

# Problem 1 — Temporal Behavior & Next-Match Performance

## Research question

> How are recent competitive volume, session depth, and post-loss requeue timing associated with performance in a player's subsequent Ranked Solo/Duo match?

## Inputs

```text
data/analysis/timelines/solo420_targets/{NA,KR,EU}.parquet
```

These timelines are created by `01_prepare_data.py` from authoritative tracked-player linkage and contain target-match outcomes plus strictly pre-target history features.

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

Session depth is the game number inside the observed session, capped for modeling/plotting at `8+`.

### H2 — Post-loss requeue timing

The analysis conditions on the previous ranked game being a loss and at least ten minutes long. Requeue delay is examined categorically and with a continuous log sensitivity specification.

### H3 — Recent ranked volume

Primary activity window:

```text
6 hours
```

Sensitivity:

```text
3 / 12 / 24 hours
```

Recent-game counts are capped at `6+` for the primary categorical model/plot.

## Inference

Primary inference uses a within-player linear probability model of target victory with:
- player fixed effects;
- historical/patch controls;
- two-way clustered uncertainty by player and physical match;
- Holm family-wise correction for the predefined behavior-effect families.

Interpretation is in **percentage-point change in next-match win probability**.

## Prediction

Three entropy-tree feature sets:

1. historical/context features;
2. behavioral features;
3. combined history + behavior.

Chronological split within region:

```text
70% train / 15% validation / 15% test
```

Validation chooses `max_depth` and `min_samples_leaf`. Final evaluation reports accuracy, precision, recall, F1, confusion matrix, ROC and ROC-AUC alongside simple baselines.

## Robustness

- authoritative vs alias-confirmed cohort;
- target-duration threshold >=10 min vs >=5 min;
- alternative session boundaries;
- alternative recent-volume windows.

## Report figures

| Figure | Generated file | Purpose |
| --- | --- | --- |
| 1 | `figure_1_inter_match_gap_ecdf.png` | justify session-threshold sensitivity |
| 2 | `figure_2_adjusted_behavior_effects.png` | adjusted H1/H2/H3 estimates + 95% CIs |
| 3 | `figure_3_prediction_evaluation.png` | held-out ROC + normalized confusion matrix |
| 4 | `figure_4_feature_importance.png` | interpret combined decision tree |
| 5 | `figure_5_tree_top_levels.png` | readable top levels of entropy tree |

## Main conclusion

There is little evidence for a universal short-term fatigue or immediate post-loss requeue penalty. Estimated behavioral effects are small/unstable across regions and parameterizations, and behavioral timing adds only limited predictive value beyond historical performance.

**Interpretation boundary:** this is observational behavioral timing, not a direct measurement of fatigue or tilt.

---

# Problem 2 — Champion Pairings & Combo Performance

## Research question

> Which champion pairs are selected together more often than expected, and how is co-selection strength related to pair performance?

## Inputs

Canonical regional participant Parquet:

```text
data/processed/full_{na,kr,eu}/participants/*.parquet
```

Filtering:
- queue 420;
- duration >=10 minutes;
- exactly five distinct champions per team;
- consistent team result.

## Analytical unit

One **unordered champion pair** within one valid team. Every five-champion team contributes ten pair observations before aggregation.

## Measures

### Raw co-pick frequency

```text
games_together
```

Answers: *Which pairs are simply seen together most often?*

### Normalized association

```text
lift = observed co-picks / expected co-picks from individual popularity
association = log2(lift)
```

Answers: *Which pairs are selected together disproportionately often after accounting for how popular each champion is?*

### Pair win surplus

```text
100 × [pair win rate − mean(individual champion win rates)]
```

Answers: *Does the pair's observed outcome sit above or below a simple individual-strength baseline?*

It is descriptive, not causal.

## Support rules

- at least **500** games together for the combo landscape;
- at least **1000** games together for high-support ranking/performance displays.

## Report figures

| Figure | Generated file | Purpose |
| --- | --- | --- |
| 6 | `figure_6_pair_rankings.png` | raw frequency versus popularity-normalized association |
| 7 | `figure_7_combo_landscape.png` | association versus descriptive win surplus, colored by support |

Supplementary:

```text
pair_win_surplus.png
```

## Main conclusion

Raw popularity and normalized co-selection capture different properties. Some pairs are unusually likely to appear together even after popularity normalization, but strong association does not consistently imply higher pair win surplus.

**Interpretation boundary:** call this **co-selection/association**, not proven champion synergy.

---

# Problem 3 — Champion Network Structure & Team-Composition Communities

## Research question

> What larger structural patterns emerge from the champion co-selection network, and which champions and communities occupy central roles within team compositions?

## Inputs

Problem 3 reuses the compact tables produced by Problem 2:

```text
data/analysis/q2/tables/champion_stats.csv
data/analysis/q2/tables/pair_stats.csv
data/analysis/q2/tables/role_counts.csv
```

This keeps the graph script fast and makes the dependency between the two problems explicit.

## Primary association graph

- node = champion;
- edge retained when pair support >=500 and `log2(lift) > 0`;
- edge weight = positive normalized association.

The raw-frequency graph is retained as a popularity-oriented comparison.

## Main graph methods

### Louvain + modularity

Louvain identifies densely connected groups on the largest supported association component. Modularity provides a structural score for the partition.

The role-composition heatmap then asks whether these purely network-derived communities have an interpretable relationship with observed Top/Jungle/Mid/ADC/Support appearances.

### Weighted PageRank

Association-weighted PageRank identifies champions connected to other structurally important champions through above-expected co-selection relationships.

The report uses normalized-association PageRank rather than raw-frequency PageRank because it better separates network position from simple popularity.

### Maximal cliques

Clique analysis first keeps the strongest 20% of association edges, then searches for maximal groups of size >=3 where every champion is connected to every other champion.

These are higher-order composition motifs, not claims that all members causally improve one another.

### Clique percolation

3-clique percolation provides an overlapping-community view where tightly connected groups can share champions.

## Comparison methods

The project also evaluates:
- Girvan-Newman community detection;
- K-Means++ on standardized co-pick profiles;
- elbow/SSE and silhouette validation for `k=2..6`;
- Ward hierarchical clustering and dendrogram;
- DBSCAN with data-driven epsilon;
- PCA visualization;
- cosine similarity of champion co-pick profiles.

These are **comparison/supplementary methods**. Louvain is the primary network partition used in the report.

## Report figures

| Figure | Generated file | Purpose |
| --- | --- | --- |
| 8 | `figure_8_louvain_community_network.png` | visual community structure |
| 9 | `figure_9_community_role_heatmap.png` | interpret communities through role composition |
| 10 | `figure_10_pagerank_association.png` | most central champions in normalized network |
| 11 | `figure_11_strongest_cliques.png` | strongest higher-order tightly connected groups |

Important supplementary figures include:
- normalized-association heatmap;
- clique-percolation sizes;
- K-Means elbow and silhouette plots;
- K-Means / hierarchical / DBSCAN PCA views;
- hierarchical dendrogram;
- Girvan-Newman network;
- raw-frequency PageRank;
- co-pick-profile similarity ranking.

## Main conclusion

The normalized champion network contains interpretable role-related communities, structurally central champions, and recurring higher-order groups. Ordinary clustering captures broader/coarser profile divisions, while graph methods better preserve the relational structure of team composition.

**Interpretation boundary:** network communities reflect co-selection structure and role complementarity; they do not prove strategic intent or causal synergy.

---

# Report-to-code figure map

```text
Problem 1 -> Figures 1-5 -> data/analysis/q1/figures/report/
Problem 2 -> Figures 6-7 -> data/analysis/q2/figures/report/
Problem 3 -> Figures 8-11 -> data/analysis/q3/figures/report/
```

All additional figures generated by the scripts are deliberately separated into `figures/supplementary/` so the final report can remain selective while the repository still demonstrates the breadth of the analysis.
