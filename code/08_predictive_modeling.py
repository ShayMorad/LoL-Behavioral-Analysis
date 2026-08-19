#!/usr/bin/env python3
r"""
08_predictive_modeling.py

Course-aligned predictive experiment for the main behavioral/temporal question.

QUESTION
--------
Do temporal-behavioral features (session depth, recent ranked volume and
requeue timing) add predictive value for a player's subsequent Solo/Duo win
beyond ordinary pre-target history?

DESIGN
------
Target:
    target_win for queue 420 target matches lasting >=10 minutes.

Ground truth:
    Riot Match-V5 target-match win/loss label.

Split:
    Chronological 70% train / 15% validation / 15% test WITHIN EACH REGION.
    The split is based only on target_start_ms, so later matches are never used
    to train models evaluated on earlier matches.

Baselines:
    1. Majority/base-rate predictor.
    2. Rolling historical win-rate predictor.

Course model:
    DecisionTreeClassifier(criterion="entropy") so splitting is based on
    entropy/information gain, matching the decision-tree material taught in
    class.

Feature groups:
    - history_only
    - behavior_only
    - combined = history + behavior

Tree complexity:
    max_depth and min_samples_leaf are selected on the validation set.
    This is pre-pruning: test data are not used for parameter selection.

Evaluation:
    Accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
    We also compare combined-vs-history AUC directly to quantify incremental
    behavioral predictive value.

Leakage prevention:
    No target-match kills, deaths, gold, damage, duration-as-feature, or final
    outcome-derived statistics are used as predictors. Only target context
    known independently of the outcome (region/patch) and strictly pre-target
    history/behavior are used.

Outputs
-------
data/analysis/prediction/
├── tables/
│   ├── split_summary.csv
│   ├── validation_grid.csv
│   ├── test_metrics.csv
│   ├── incremental_value.csv
│   ├── confusion_matrices.csv
│   ├── feature_importance.csv
│   └── error_analysis.csv
├── figures/
│   ├── roc_curves_overall.png
│   ├── test_auc_comparison.png
│   ├── combined_feature_importance.png
│   └── combined_tree_top_levels.png
├── predictions/
│   └── test_predictions.parquet
└── audit/
    └── predictive_modeling_summary.json

This is predictive evaluation, not a causal analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.tree import DecisionTreeClassifier, plot_tree
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for 08_predictive_modeling.py.\n"
        "Install it in the project environment with:\n"
        "    pip install scikit-learn"
    ) from exc


RANDOM_STATE = 67978

# Pre-pruning grid. Validation data choose among these settings.
MAX_DEPTH_GRID = (2, 3, 4, 5, 6, 8)
MIN_SAMPLES_LEAF_GRID = (250, 1000, 3000)

SESSION_CAP = 8
VOLUME_CAP = 6

REGIONS = ("NA", "KR", "EU")

HISTORY_BASE_FEATURES = [
    "prior_ranked_win_rate",
    "log1p_prior_ranked_matches",
    "prev_ranked_win",
    "log1p_prev_ranked_kda",
    "prev_ranked_cs_per_min",
    "prev_ranked_gold_per_min",
    "prev_ranked_damage_per_min",
    "prev_ranked_vision_per_min",
    "log1p_prev_ranked_loss_streak",
    "log1p_prev_ranked_win_streak",
    "prev_queue_was_flex440",
    "log1p_prev_ranked_duration_min",
    "prior_ranked_mean_kda",
    "prior_ranked_mean_cs_per_min",
    "prior_ranked_mean_gold_per_min",
    "prior_ranked_mean_damage_per_min",
]

BEHAVIOR_BASE_FEATURES = [
    "log1p_gap_from_prev_ranked_min",
    "session_depth_30m",
    "ranked_games_prev_6h_capped",
    "ranked_minutes_prev_6h",
    "previous_ranked_was_loss",
    "log1p_post_loss_gap_min",
    "champion_changed_from_prev_ranked",
    "role_changed_from_prev_ranked",
]

COMMON_CONTEXT_PREFIXES = ("region_", "patch_")


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--timelines",
        type=Path,
        required=True,
        help="Folder containing NA/KR/EU solo420 target Parquet files.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duckdb-memory-limit", default="4GB")
    p.add_argument("--duckdb-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def timeline_glob(folder: Path) -> str:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in: {folder}")
    return sql_path(folder / "*.parquet")


def load_primary_sample(
    con: duckdb.DuckDBPyConnection,
    folder: Path,
) -> pd.DataFrame:
    glob = timeline_glob(folder)

    required = {
        "source",
        "player_id",
        "match_id",
        "target_win",
        "target_patch",
        "target_start_ms",
        "target_duration_s",
        "has_prior_ranked_match",
        "prior_ranked_matches",
        "prior_ranked_win_rate",
        "prev_ranked_win",
        "prev_ranked_kda",
        "prev_ranked_cs_per_min",
        "prev_ranked_gold_per_min",
        "prev_ranked_damage_per_min",
        "prev_ranked_vision_per_min",
        "prev_ranked_loss_streak",
        "prev_ranked_win_streak",
        "prev_ranked_queue_id",
        "prev_ranked_duration_s",
        "prior_ranked_mean_kda",
        "prior_ranked_mean_cs_per_min",
        "prior_ranked_mean_gold_per_min",
        "prior_ranked_mean_damage_per_min",
        "gap_from_prev_ranked_min",
        "post_loss_ranked_requeue_gap_min",
        "ranked_session_game_no_30m",
        "ranked_games_prev_6h",
        "ranked_minutes_played_prev_6h",
        "champion_changed_from_prev_ranked",
        "role_changed_from_prev_ranked",
    }

    available = set(
        con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{glob}')"
        )
        .fetchdf()["column_name"]
        .astype(str)
    )
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Timeline input missing required columns: {missing}")

    cols = ", ".join(sorted(required))

    return con.execute(
        f"""
        SELECT {cols}
        FROM read_parquet('{glob}')
        WHERE has_prior_ranked_match
          AND target_win IS NOT NULL
          AND target_duration_s >= 600
        """
    ).fetchdf()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["prior_ranked_win_rate"] = pd.to_numeric(
        out["prior_ranked_win_rate"], errors="coerce"
    )

    out["log1p_prior_ranked_matches"] = np.log1p(
        pd.to_numeric(out["prior_ranked_matches"], errors="coerce").clip(lower=0)
    )
    out["prev_ranked_win"] = out["prev_ranked_win"].astype(float)
    out["log1p_prev_ranked_kda"] = np.log1p(
        pd.to_numeric(out["prev_ranked_kda"], errors="coerce").clip(lower=0)
    )
    out["log1p_prev_ranked_loss_streak"] = np.log1p(
        pd.to_numeric(out["prev_ranked_loss_streak"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    out["log1p_prev_ranked_win_streak"] = np.log1p(
        pd.to_numeric(out["prev_ranked_win_streak"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    out["prev_queue_was_flex440"] = (
        out["prev_ranked_queue_id"] == 440
    ).astype(float)
    out["log1p_prev_ranked_duration_min"] = np.log1p(
        pd.to_numeric(out["prev_ranked_duration_s"], errors="coerce")
        .clip(lower=0)
        / 60.0
    )

    out["log1p_gap_from_prev_ranked_min"] = np.log1p(
        pd.to_numeric(out["gap_from_prev_ranked_min"], errors="coerce")
        .clip(lower=0)
    )
    out["session_depth_30m"] = (
        pd.to_numeric(out["ranked_session_game_no_30m"], errors="coerce")
        .clip(lower=1, upper=SESSION_CAP)
    )
    out["ranked_games_prev_6h_capped"] = (
        pd.to_numeric(out["ranked_games_prev_6h"], errors="coerce")
        .clip(lower=0, upper=VOLUME_CAP)
    )
    out["ranked_minutes_prev_6h"] = pd.to_numeric(
        out["ranked_minutes_played_prev_6h"], errors="coerce"
    ).clip(lower=0)

    out["previous_ranked_was_loss"] = (
        out["prev_ranked_win"] == 0
    ).astype(float)

    raw_post_loss_gap = pd.to_numeric(
        out["post_loss_ranked_requeue_gap_min"], errors="coerce"
    )
    out["log1p_post_loss_gap_min"] = np.where(
        out["previous_ranked_was_loss"] == 1,
        np.log1p(raw_post_loss_gap.clip(lower=0).fillna(0)),
        0.0,
    )

    for col in (
        "champion_changed_from_prev_ranked",
        "role_changed_from_prev_ranked",
    ):
        out[col] = out[col].astype(float)

    out["target_win"] = out["target_win"].astype(int)
    out["source"] = out["source"].astype(str)
    out["target_patch"] = out["target_patch"].astype(str)

    return out


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return data with split labels plus a compact per-region threshold table.
    """
    out = df.copy()
    threshold_rows = []

    split = pd.Series(index=out.index, dtype="object")

    for source in REGIONS:
        mask = out["source"] == source
        times = out.loc[mask, "target_start_ms"].astype(np.int64)

        q70 = float(times.quantile(0.70))
        q85 = float(times.quantile(0.85))

        split.loc[mask & (out["target_start_ms"] < q70)] = "train"
        split.loc[
            mask
            & (out["target_start_ms"] >= q70)
            & (out["target_start_ms"] < q85)
        ] = "validation"
        split.loc[
            mask & (out["target_start_ms"] >= q85)
        ] = "test"

        threshold_rows.append(
            {
                "source": source,
                "train_validation_boundary_ms": int(q70),
                "validation_test_boundary_ms": int(q85),
            }
        )

    out["split"] = split

    if out["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a chronological split.")

    # Same physical match must never cross splits.
    cross_split = (
        out.groupby(["source", "match_id"])["split"]
        .nunique()
        .gt(1)
        .sum()
    )
    if cross_split:
        raise RuntimeError(
            f"{cross_split} physical matches cross chronological split boundaries."
        )

    return out, pd.DataFrame(threshold_rows)


def fit_context_schema(train: pd.DataFrame) -> dict:
    return {
        "regions": sorted(train["source"].dropna().unique().tolist()),
        "patches": sorted(train["target_patch"].dropna().unique().tolist()),
    }


def make_context_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Use train-defined categories only. Unknown future patches/regions become
    # all-zero for that category family rather than leaking test schema.
    for region in schema["regions"][1:]:
        out[f"region_{region}"] = (df["source"] == region).astype(float)

    for patch in schema["patches"][1:]:
        out[f"patch_{patch}"] = (df["target_patch"] == patch).astype(float)

    return out


def fill_train_medians(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    medians = {}
    train_f = train.copy()
    val_f = validation.copy()
    test_f = test.copy()

    for col in train.columns:
        med = float(pd.to_numeric(train[col], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.0
        medians[col] = med

        train_f[col] = pd.to_numeric(
            train_f[col], errors="coerce"
        ).fillna(med)
        val_f[col] = pd.to_numeric(
            val_f[col], errors="coerce"
        ).fillna(med)
        test_f[col] = pd.to_numeric(
            test_f[col], errors="coerce"
        ).fillna(med)

    return train_f, val_f, test_f, medians


def metrics_row(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    model_name: str,
    subgroup: str,
) -> dict:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    return {
        "model": model_name,
        "subgroup": subgroup,
        "n": int(len(y_true)),
        "observed_win_rate": float(np.mean(y_true)),
        "mean_predicted_win_probability": float(np.mean(prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_probabilities(
    frame: pd.DataFrame,
    prob: np.ndarray,
    model_name: str,
) -> list[dict]:
    rows = []
    y = frame["target_win"].to_numpy(dtype=int)

    rows.append(
        metrics_row(y, prob, model_name=model_name, subgroup="ALL")
    )

    for source in REGIONS:
        mask = frame["source"].to_numpy() == source
        if mask.any():
            rows.append(
                metrics_row(
                    y[mask],
                    prob[mask],
                    model_name=model_name,
                    subgroup=source,
                )
            )

    return rows


def decision_tree_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set: str,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, dict]:
    rows = []
    best = None
    best_auc = -np.inf

    for depth in MAX_DEPTH_GRID:
        for min_leaf in MIN_SAMPLES_LEAF_GRID:
            model = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

            train_prob = model.predict_proba(X_train)[:, 1]
            val_prob = model.predict_proba(X_val)[:, 1]

            train_auc = roc_auc_score(y_train, train_prob)
            val_auc = roc_auc_score(y_val, val_prob)

            rows.append(
                {
                    "feature_set": feature_set,
                    "max_depth": depth,
                    "min_samples_leaf": min_leaf,
                    "train_auc": train_auc,
                    "validation_auc": val_auc,
                    "validation_minus_train_auc": val_auc - train_auc,
                    "nodes": model.tree_.node_count,
                    "leaves": model.tree_.n_leaves,
                }
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best = model

    grid = pd.DataFrame(rows)
    best_row = (
        grid.sort_values(
            ["validation_auc", "max_depth", "min_samples_leaf"],
            ascending=[False, True, False],
        )
        .iloc[0]
        .to_dict()
    )

    if best is None:
        raise RuntimeError("Decision-tree grid produced no model.")

    # Recreate the exact best model from the sorted table to avoid any
    # ambiguity from traversal order.
    best_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=int(best_row["max_depth"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_train, y_train)

    return best_model, grid, best_row


def save_roc_figure(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    output: Path,
) -> None:
    y = test_df["target_win"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for name in (
        "historical_win_rate_baseline",
        "history_tree",
        "behavior_tree",
        "combined_tree",
    ):
        prob = predictions[name]
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Chronological held-out test ROC curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_auc_comparison(metrics: pd.DataFrame, output: Path) -> None:
    sub = metrics[
        (metrics["subgroup"] == "ALL")
        & metrics["model"].isin(
            [
                "majority_baseline",
                "historical_win_rate_baseline",
                "history_tree",
                "behavior_tree",
                "combined_tree",
            ]
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["roc_auc"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(sub["model"], rotation=25, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Held-out predictive performance by model")
    ax.axhline(0.5, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    df.to_csv(output_csv, index=False)

    top = df.head(15).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Normalized decision-tree feature importance")
    ax.set_title("Combined tree: top entropy/information-gain features")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    return df


def save_tree_figure(
    model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=["Loss", "Win"],
        max_depth=3,
        filled=False,
        rounded=True,
        proportion=True,
        impurity=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(
        "Combined entropy-based decision tree (first 3 levels shown)"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def error_analysis(
    test: pd.DataFrame,
    prob: np.ndarray,
) -> pd.DataFrame:
    work = test.copy()
    work["pred_prob"] = prob
    work["pred"] = (prob >= 0.5).astype(int)
    work["correct"] = (
        work["pred"].to_numpy() == work["target_win"].to_numpy()
    )

    work["session_depth_bucket"] = (
        work["session_depth_30m"]
        .clip(upper=5)
        .map(lambda x: "5+" if x >= 5 else str(int(x)))
    )
    work["volume_bucket"] = (
        work["ranked_games_prev_6h_capped"]
        .clip(upper=4)
        .map(lambda x: "4+" if x >= 4 else str(int(x)))
    )
    work["previous_result"] = np.where(
        work["previous_ranked_was_loss"] == 1, "loss", "win"
    )

    frames = []

    for group_name, cols in [
        ("region", ["source"]),
        ("session_depth", ["source", "session_depth_bucket"]),
        ("recent_volume", ["source", "volume_bucket"]),
        ("previous_result", ["source", "previous_result"]),
    ]:
        rows = []
        for keys, sub in work.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            tn, fp, fn, tp = confusion_matrix(
                sub["target_win"],
                sub["pred"],
                labels=[0, 1],
            ).ravel()

            row = {
                "group_type": group_name,
                "n": len(sub),
                "observed_win_rate": sub["target_win"].mean(),
                "mean_predicted_win_probability": sub["pred_prob"].mean(),
                "accuracy": sub["correct"].mean(),
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
            }
            for c, k in zip(cols, keys):
                row[c] = k
            rows.append(row)

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    prepare_dir(args.output, args.overwrite)

    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    pred_dir = args.output / "predictions"
    audit_dir = args.output / "audit"

    for d in (table_dir, figure_dir, pred_dir, audit_dir):
        d.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{args.duckdb_memory_limit}'")
    con.execute(f"SET threads={int(args.duckdb_threads)}")

    try:
        print("[prediction] loading primary timeline sample...", flush=True)
        raw = load_primary_sample(con, args.timelines)
    finally:
        con.close()

    data = add_engineered_features(raw)
    del raw

    data, split_thresholds = chronological_split(data)

    split_summary = (
        data.groupby(["source", "split"])
        .agg(
            rows=("target_win", "size"),
            players=("player_id", "nunique"),
            physical_matches=("match_id", "nunique"),
            win_rate=("target_win", "mean"),
            min_time_ms=("target_start_ms", "min"),
            max_time_ms=("target_start_ms", "max"),
        )
        .reset_index()
    )
    split_summary = split_summary.merge(
        split_thresholds, on="source", how="left"
    )
    split_summary.to_csv(table_dir / "split_summary.csv", index=False)

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    print(
        f"[prediction] train={len(train):,}, "
        f"validation={len(validation):,}, test={len(test):,}",
        flush=True,
    )

    context_schema = fit_context_schema(train)
    train_context = make_context_features(train, context_schema)
    val_context = make_context_features(validation, context_schema)
    test_context = make_context_features(test, context_schema)

    history_features = HISTORY_BASE_FEATURES + list(train_context.columns)
    behavior_features = BEHAVIOR_BASE_FEATURES + list(train_context.columns)
    combined_features = (
        HISTORY_BASE_FEATURES
        + BEHAVIOR_BASE_FEATURES
        + list(train_context.columns)
    )

    # Assemble feature frames.
    train_full = pd.concat(
        [
            train[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES],
            train_context,
        ],
        axis=1,
    )
    val_full = pd.concat(
        [
            validation[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES],
            val_context,
        ],
        axis=1,
    )
    test_full = pd.concat(
        [
            test[HISTORY_BASE_FEATURES + BEHAVIOR_BASE_FEATURES],
            test_context,
        ],
        axis=1,
    )

    train_full, val_full, test_full, medians = fill_train_medians(
        train_full, val_full, test_full
    )

    y_train = train["target_win"].astype(int)
    y_val = validation["target_win"].astype(int)
    y_test = test["target_win"].astype(int)

    feature_sets = {
        "history_tree": history_features,
        "behavior_tree": behavior_features,
        "combined_tree": combined_features,
    }

    validation_frames = []
    models = {}
    best_params = {}

    for model_name, features in feature_sets.items():
        print(f"[prediction] tuning {model_name}...", flush=True)

        model, grid, best = decision_tree_grid(
            train_full[features],
            y_train,
            val_full[features],
            y_val,
            feature_set=model_name,
        )
        validation_frames.append(grid)
        models[model_name] = model
        best_params[model_name] = best

        print(
            f"  best max_depth={int(best['max_depth'])}, "
            f"min_samples_leaf={int(best['min_samples_leaf'])}, "
            f"validation AUC={best['validation_auc']:.5f}",
            flush=True,
        )

    validation_grid = pd.concat(validation_frames, ignore_index=True)
    validation_grid.to_csv(
        table_dir / "validation_grid.csv", index=False
    )

    # ------------------------------------------------------------------
    # Test predictions
    # ------------------------------------------------------------------
    predictions: Dict[str, np.ndarray] = {}

    train_base_rate = float(y_train.mean())
    predictions["majority_baseline"] = np.full(
        len(test), train_base_rate, dtype=float
    )

    predictions["historical_win_rate_baseline"] = (
        pd.to_numeric(
            test["prior_ranked_win_rate"], errors="coerce"
        )
        .fillna(train_base_rate)
        .clip(0.001, 0.999)
        .to_numpy(dtype=float)
    )

    for model_name, model in models.items():
        features = feature_sets[model_name]
        predictions[model_name] = model.predict_proba(
            test_full[features]
        )[:, 1]

    metric_rows = []
    for model_name, prob in predictions.items():
        metric_rows.extend(
            evaluate_probabilities(test, prob, model_name)
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_dir / "test_metrics.csv", index=False)

    # Confusion matrix in long form.
    confusion_cols = [
        "model",
        "subgroup",
        "n",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    metrics[confusion_cols].to_csv(
        table_dir / "confusion_matrices.csv", index=False
    )

    # Incremental behavioral predictive value.
    incremental_rows = []
    for subgroup in ["ALL", *REGIONS]:
        h = metrics[
            (metrics["model"] == "history_tree")
            & (metrics["subgroup"] == subgroup)
        ].iloc[0]
        c = metrics[
            (metrics["model"] == "combined_tree")
            & (metrics["subgroup"] == subgroup)
        ].iloc[0]

        incremental_rows.append(
            {
                "subgroup": subgroup,
                "history_auc": h["roc_auc"],
                "combined_auc": c["roc_auc"],
                "delta_auc_combined_minus_history": (
                    c["roc_auc"] - h["roc_auc"]
                ),
                "history_accuracy": h["accuracy"],
                "combined_accuracy": c["accuracy"],
                "delta_accuracy_combined_minus_history": (
                    c["accuracy"] - h["accuracy"]
                ),
                "history_f1": h["f1"],
                "combined_f1": c["f1"],
                "delta_f1_combined_minus_history": (
                    c["f1"] - h["f1"]
                ),
            }
        )

    incremental = pd.DataFrame(incremental_rows)
    incremental.to_csv(
        table_dir / "incremental_value.csv", index=False
    )

    # ------------------------------------------------------------------
    # Feature importance and error analysis
    # ------------------------------------------------------------------
    combined_features = feature_sets["combined_tree"]
    importance = save_feature_importance(
        models["combined_tree"],
        combined_features,
        table_dir / "feature_importance.csv",
        figure_dir / "combined_feature_importance.png",
    )

    errors = error_analysis(
        test,
        predictions["combined_tree"],
    )
    errors.to_csv(table_dir / "error_analysis.csv", index=False)

    # Save compact test predictions for reproducible later error analysis.
    pred_out = test[
        [
            "source",
            "player_id",
            "match_id",
            "target_start_ms",
            "target_win",
            "session_depth_30m",
            "ranked_games_prev_6h_capped",
            "previous_ranked_was_loss",
        ]
    ].copy()

    for name, prob in predictions.items():
        pred_out[f"prob_{name}"] = prob

    pred_out.to_parquet(
        pred_dir / "test_predictions.parquet",
        index=False,
        compression="zstd",
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    save_roc_figure(
        test,
        predictions,
        figure_dir / "roc_curves_overall.png",
    )
    save_auc_comparison(
        metrics,
        figure_dir / "test_auc_comparison.png",
    )
    save_tree_figure(
        models["combined_tree"],
        combined_features,
        figure_dir / "combined_tree_top_levels.png",
    )

    payload = {
        "target": "target_win",
        "sample": (
            "Queue 420 targets with prior ranked history and target duration >=10m."
        ),
        "split": (
            "Chronological 70% train / 15% validation / 15% test within region."
        ),
        "tree_criterion": "entropy",
        "tree_parameter_grid": {
            "max_depth": list(MAX_DEPTH_GRID),
            "min_samples_leaf": list(MIN_SAMPLES_LEAF_GRID),
        },
        "best_parameters": best_params,
        "baselines": [
            "majority/base-rate",
            "rolling historical win rate",
        ],
        "feature_sets": {
            "history_tree": history_features,
            "behavior_tree": behavior_features,
            "combined_tree": combined_features,
        },
        "metrics": [
            "accuracy",
            "precision",
            "recall",
            "F1",
            "ROC-AUC",
            "confusion matrix",
        ],
        "leakage_policy": (
            "No target-match performance or outcome-derived predictors."
        ),
        "train_base_win_rate": train_base_rate,
        "incremental_value": incremental_rows,
        "top_combined_features": importance.head(15).to_dict("records"),
    }

    (audit_dir / "predictive_modeling_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nPREDICTIVE MODELING COMPLETE\n")
    print(
        metrics[
            [
                "model",
                "subgroup",
                "n",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    print("\nINCREMENTAL BEHAVIORAL VALUE\n")
    print(incremental.to_string(index=False))

    print(f"\nTables:      {table_dir}")
    print(f"Figures:     {figure_dir}")
    print(f"Predictions: {pred_dir}")



if __name__ == "__main__":
    main()
