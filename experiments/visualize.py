#!/usr/bin/env python3
"""
Visualization helpers for synthetic Jira experiments.

- Feedback bias plots
- Side-by-side comparison: baseline vs feedback-aware model

Usage (with uv):
  uv run --with pandas==2.2.2 --with numpy==1.26.4 --with scikit-learn==1.5.1 \
         --with joblib==1.4.2 --with matplotlib==3.8.4 --with seaborn==0.13.2 \
         python experiments/visualize.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None  # type: ignore
    from sklearn.metrics import mean_squared_error

DATA_DIR = Path("experiments/jira_data")
PLOTS_DIR = Path("experiments/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load(issues_csv: Path, events_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    issues = pd.read_csv(issues_csv)
    events = pd.read_csv(events_csv)
    return issues, events


def build_features(issues: pd.DataFrame, events: pd.DataFrame):
    ev = events.groupby("issue_key").agg(
        events_count=("event_index", "count"),
        total_minutes_in_states=("minutes_in_from_status", "sum"),
        review_transitions=("to_status", lambda s: (s == "In Review").sum()),
        qa_transitions=("to_status", lambda s: (s == "QA").sum()),
        reopen_transitions=("to_status", lambda s: (s == "Reopened").sum()),
        comments_sum=("comments_delta", "sum"),
        prs_sum=("prs_delta", "sum"),
    )
    df = issues.merge(ev, left_on="issue_key", right_index=True, how="left")
    for c in [
        "events_count",
        "total_minutes_in_states",
        "review_transitions",
        "qa_transitions",
        "reopen_transitions",
        "comments_sum",
        "prs_sum",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    y = df["resolve_duration_hours"].astype(float)
    numeric = [
        "time_to_first_response_hours",
        "time_to_in_progress_hours",
        "assignee_experience_years",
        "num_comments",
        "num_watchers",
        "num_attachments",
        "num_linked_issues",
        "reopened_count",
        "pull_requests_linked",
        "events_count",
        "total_minutes_in_states",
        "review_transitions",
        "qa_transitions",
        "reopen_transitions",
        "comments_sum",
        "prs_sum",
    ]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [
        "issue_type",
        "priority",
        "severity",
        "environment",
        "affected_component",
        "customer_impact",
        "ci_cd_status",
        "reporter_team",
        "assignee_team",
    ]
    categorical = [c for c in categorical if c in df.columns]
    if "title" in df.columns:
        df["title_len"] = df["title"].astype(str).str.len()
    if "context" in df.columns:
        df["context_len"] = df["context"].astype(str).str.len()
    numeric += [c for c in ["title_len", "context_len"] if c in df.columns]
    X = df[numeric + categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    # Carry along the merged DataFrame for per-issue feature rows
    df_feats = df[numeric + categorical].copy()
    df_feats.insert(0, "issue_key", df["issue_key"])  # keep key for lookup
    return X, y, preprocessor, df_feats


def train_baseline(X, y, preproc, seed=42, test_size: float = 0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = Pipeline(
        [
            ("prep", preproc),
            (
                "rf",
                RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            ),
        ]
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return model, y_te, preds


def train_feedback_weighted(
    X,
    y,
    preproc,
    issues: pd.DataFrame,
    feedback: pd.DataFrame,
    seed=42,
    test_size: float = 0.2,
):
    # Create per-sample weights: downweight disliked samples modestly
    fb = feedback.copy()
    fb["feedback_type"] = fb["feedback_type"].str.lower()
    verdict = fb.groupby("issue_key")["feedback_type"].agg(
        lambda s: (s == "like").mean()
    )
    # Merge back to issues
    weights_df = issues[["issue_key"]].merge(
        verdict.rename("like_rate"), left_on="issue_key", right_index=True, how="left"
    )
    # Default weight 1.0, disliked-heavy get lower weight
    w = weights_df["like_rate"].fillna(0.6).clip(0.2, 1.2)
    sample_weight = w.values

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, sample_weight, test_size=test_size, random_state=seed
    )

    model = Pipeline(
        [
            ("prep", preproc),
            (
                "rf",
                RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
            ),
        ]
    )
    model.fit(X_tr, y_tr, rf__sample_weight=w_tr)
    preds = model.predict(X_te)
    return model, y_te, preds


def plot_comparison(y_true_a, preds_a, y_true_b, preds_b, out_path: Path):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_true_a, y=preds_a, s=18, alpha=0.5, label="Baseline")
    sns.scatterplot(x=y_true_b, y=preds_b, s=18, alpha=0.5, label="Feedback-weighted")
    lim = (0, max(float(np.max(y_true_a)), float(np.max(y_true_b))) * 1.05)
    plt.plot(lim, lim, "k--", linewidth=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("Actual resolve_duration_hours")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual: Baseline vs Feedback-weighted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    # Error comparison bar
    mae_a = mean_absolute_error(y_true_a, preds_a)
    if root_mean_squared_error is not None:
        rmse_a = root_mean_squared_error(y_true_a, preds_a)
    else:
        rmse_a = mean_squared_error(y_true_a, preds_a, squared=False)
    mae_b = mean_absolute_error(y_true_b, preds_b)
    if root_mean_squared_error is not None:
        rmse_b = root_mean_squared_error(y_true_b, preds_b)
    else:
        rmse_b = mean_squared_error(y_true_b, preds_b, squared=False)

    plt.figure(figsize=(6, 4))
    metrics = pd.DataFrame(
        {
            "Model": ["Baseline", "Feedback-weighted"],
            "MAE": [mae_a, mae_b],
            "RMSE": [rmse_a, rmse_b],
        }
    )
    metrics = metrics.melt("Model", var_name="Metric", value_name="Value")
    sns.barplot(data=metrics, x="Metric", y="Value", hue="Model")
    plt.title("Error comparison")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_errors.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization utilities")
    parser.add_argument("--issues", type=str, default=str(DATA_DIR / "synthetic.csv"))
    parser.add_argument(
        "--events", type=str, default=str(DATA_DIR / "synthetic_events.csv")
    )
    parser.add_argument(
        "--feedback", type=str, default=str(DATA_DIR / "synthetic_feedback.csv")
    )
    parser.add_argument("--plots-dir", type=str, default=str(PLOTS_DIR))
    # Optional external split indices
    parser.add_argument(
        "--train-index",
        type=str,
        help="Path to CSV with train index (must contain issue_key column)",
    )
    parser.add_argument(
        "--test-index",
        type=str,
        help="Path to CSV with test index (must contain issue_key column)",
    )
    parser.add_argument(
        "--split-index",
        type=str,
        help="Path to CSV with a single combined split (columns: issue_key,test)",
    )

    # Event timeline options
    parser.add_argument(
        "--plot-events", action="store_true", help="Render per-issue event timelines"
    )
    parser.add_argument(
        "--issue-key",
        action="append",
        help="Specific issue_key(s) to plot; can be passed multiple times",
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="If >0, randomly sample N issues to plot"
    )
    parser.add_argument(
        "--all", action="store_true", help="Plot events for all issues (may be slow)"
    )
    parser.add_argument(
        "--annotate-pred",
        action="store_true",
        help="Annotate plots with predicted resolve time and print predictions",
    )
    # Train/test control for sampling and annotation
    parser.add_argument(
        "--only-test",
        action="store_true",
        help="When sampling/plotting, use only items from the test split",
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="When sampling/plotting, use only items from the train split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for reproducible split (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split",
    )

    args = parser.parse_args()

    issues_csv = Path(args.issues)
    events_csv = Path(args.events)
    feedback_csv = Path(args.feedback)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    issues, events = load(issues_csv, events_csv)

    def plot_issue_timeline(
        issue_key: str, out_dir: Path, predicted_hours: float | None = None
    ):
        df_e = events[events["issue_key"] == issue_key].copy()
        if df_e.empty:
            return False
        df_e = (
            df_e.sort_values(["event_index", "transitioned_at"])
            if {"event_index", "transitioned_at"}.issubset(df_e.columns)
            else df_e.sort_values("event_index")
        )
        issue_row = issues[issues["issue_key"] == issue_key].iloc[0]
        created = pd.to_datetime(issue_row.get("created_at"))
        resolved = pd.to_datetime(issue_row.get("resolved_at"))
        # Build step series
        times = []
        yvals = []
        statuses = [
            "Open",
            "Triaged",
            "In Progress",
            "In Review",
            "QA",
            "Done",
            "Reopened",
        ]
        status_to_y = {s: i for i, s in enumerate(statuses)}
        # Initialize current status from first status_change's from_status or 'Open'
        first_sc = df_e[df_e["event_type"] == "status_change"].head(1)
        current_status = (
            first_sc["from_status"].iloc[0]
            if not first_sc.empty
            else (
                df_e["from_status"].iloc[0] if "from_status" in df_e.columns else "Open"
            )
        )
        current_time = created
        times.append(current_time)
        yvals.append(status_to_y.get(current_status, status_to_y["Open"]))
        # Iterate events
        for _, ev in df_e.iterrows():
            t_next = pd.to_datetime(ev.get("transitioned_at", None))
            if pd.isna(t_next):
                continue
            # Horizontal segment until this event
            times.append(t_next)
            yvals.append(status_to_y.get(current_status, status_to_y["Open"]))
            # Vertical step to next status for status_change
            if ev.get("event_type") == "status_change":
                next_status = ev.get("to_status", current_status)
                times.append(t_next)
                yvals.append(status_to_y.get(next_status, status_to_y["Open"]))
                current_status = next_status
                current_time = t_next
        # Extend to resolved
        if isinstance(resolved, pd.Timestamp) and not pd.isna(resolved):
            times.append(resolved)
            yvals.append(status_to_y.get(current_status, status_to_y["Open"]))

        # Plot
        plt.figure(figsize=(9, 3.5))
        plt.step(times, yvals, where="post")
        # Mark first_response if present
        fr = df_e[df_e["event_type"] == "first_response"]
        if not fr.empty:
            fr_t = pd.to_datetime(fr["transitioned_at"].iloc[0])
            plt.axvline(
                fr_t,
                color="tab:green",
                linestyle="--",
                alpha=0.6,
                label="first_response",
            )
        # Mark predicted resolve time if provided
        if (
            predicted_hours is not None
            and isinstance(created, pd.Timestamp)
            and not pd.isna(created)
        ):
            pred_time = created + timedelta(hours=float(predicted_hours))
            plt.axvline(
                pred_time,
                color="tab:red",
                linestyle="--",
                alpha=0.7,
                label="predicted_resolve",
            )
        # Cosmetics
        plt.yticks(list(status_to_y.values()), list(status_to_y.keys()))
        plt.xlabel("Time")
        plt.title(f"Status timeline: {issue_key}")
        # Show legend only if we added labeled lines
        try:
            plt.legend(loc="best")
        except Exception:
            pass
        plt.tight_layout()
        out = out_dir / "events"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"events_{issue_key}.png"
        plt.savefig(path)
        plt.close()
        return True

    # If requested, render event timelines
    if args.plot_events:
        # Optional train/test split to filter keys
        X_all, y_all, preproc_all, df_feats_all = build_features(issues, events)
        train_keys: set[str] | None = None
        test_keys: set[str] | None = None
        # Prefer explicit split files if provided
        # Load combined split if provided
        if args.split_index:
            df_split = pd.read_csv(args.split_index)
            if not {"issue_key", "test"}.issubset(df_split.columns):
                raise SystemExit(
                    "Combined split CSV must contain 'issue_key' and 'test' columns"
                )
            df_split["test"] = df_split["test"].astype(int)
            train_keys = set(
                df_split.loc[df_split["test"] == 0, "issue_key"].astype(str)
            )
            test_keys = set(
                df_split.loc[df_split["test"] == 1, "issue_key"].astype(str)
            )
        elif args.train_index and args.test_index:
            df_tr = pd.read_csv(args.train_index)
            df_te = pd.read_csv(args.test_index)
            if "issue_key" not in df_tr.columns or "issue_key" not in df_te.columns:
                raise SystemExit("Split CSVs must contain an 'issue_key' column")
            train_keys = set(df_tr["issue_key"].astype(str).tolist())
            test_keys = set(df_te["issue_key"].astype(str).tolist())
        elif args.only_train or args.only_test:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all,
                y_all,
                test_size=args.test_size,
                random_state=args.split_seed,
            )
            # Map indices back to issue_keys using df_feats_all index alignment
            # Ensure df_feats_all aligns with X_all/y_all rows
            df_keys = df_feats_all["issue_key"].reset_index(drop=True)
            # After train_test_split, X_tr, X_te are NumPy/Pandas subsets preserving order; use y indices
            # We reconstruct indices via lengths
            # Split returns arrays; can't directly recover indices; use a stratagem: get masks
            n = len(y_all)
            # Build a boolean mask by re-splitting indices
            idx = np.arange(n)
            idx_tr, idx_te = train_test_split(
                idx, test_size=args.test_size, random_state=args.split_seed
            )
            train_keys = set(df_keys.iloc[idx_tr].tolist())
            test_keys = set(df_keys.iloc[idx_te].tolist())

        keys: list[str]
        if args.all:
            keys = issues["issue_key"].tolist()
        elif args.issue_key:
            keys = args.issue_key
        elif args.sample and args.sample > 0:
            pool = issues["issue_key"].tolist()
            if args.only_train and train_keys is not None:
                pool = [k for k in pool if k in train_keys]
            if args.only_test and test_keys is not None:
                pool = [k for k in pool if k in test_keys]
            if len(pool) == 0:
                print("No issues match the selected split; falling back to all issues.")
                pool = issues["issue_key"].tolist()
            keys = (
                pd.Series(pool)
                .sample(n=min(args.sample, len(pool)), random_state=42)
                .tolist()
            )
        else:
            pool = issues["issue_key"].tolist()
            if args.only_train and train_keys is not None:
                pool = [k for k in pool if k in train_keys]
            if args.only_test and test_keys is not None:
                pool = [k for k in pool if k in test_keys]
            if len(pool) == 0:
                pool = issues["issue_key"].tolist()
            keys = (
                pd.Series(pool).sample(n=min(10, len(pool)), random_state=42).tolist()
            )

        # Optionally train a baseline model to annotate predictions
        predictions: dict[str, float] = {}
        if args.annotate_pred:
            # Train only on the train split for fairness when annotating predictions
            if train_keys is None or test_keys is None:
                # Build split if not already created
                n = len(y_all)
                idx = np.arange(n)
                idx_tr, idx_te = train_test_split(
                    idx, test_size=args.test_size, random_state=args.split_seed
                )
                df_keys = df_feats_all["issue_key"].reset_index(drop=True)
                train_keys = set(df_keys.iloc[idx_tr].tolist())
                test_keys = set(df_keys.iloc[idx_te].tolist())
            # Build train DataFrame
            df_train = df_feats_all[df_feats_all["issue_key"].isin(train_keys)].copy()
            X_train = df_train.drop(columns=["issue_key"])
            y_series = issues.set_index("issue_key")["resolve_duration_hours"]
            y_train = y_series.loc[df_train["issue_key"].values].astype(float).values
            # Fit model on train
            model = Pipeline(
                [
                    ("prep", preproc_all),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=300, random_state=args.split_seed, n_jobs=-1
                        ),
                    ),
                ]
            )
            model.fit(X_train, y_train)
            # Predict for selected keys
            df_sel = df_feats_all[df_feats_all["issue_key"].isin(keys)].copy()
            if not df_sel.empty:
                X_sel = df_sel.drop(columns=["issue_key"])
                preds = model.predict(X_sel)
                predictions = {k: float(p) for k, p in zip(df_sel["issue_key"], preds)}

        count = 0
        for k in keys:
            ok = plot_issue_timeline(k, plots_dir, predictions.get(k))
            if ok:
                count += 1
                if args.annotate_pred and k in predictions:
                    # Print predicted vs actual
                    actual = float(
                        issues.loc[
                            issues["issue_key"] == k, "resolve_duration_hours"
                        ].iloc[0]
                    )
                    print(f"{k}: predicted={predictions[k]:.2f}h, actual={actual:.2f}h")
        print(f"Saved {count} event timeline plot(s) to {plots_dir / 'events'}")

    # Model comparison still available when not only plotting events
    if not args.plot_events or (
        args.issue_key is None and not args.all and args.sample == 0
    ):
        X, y, preproc, df_feats = build_features(issues, events)
        # If explicit split indices provided, use them for comparison too
        if args.split_index or (args.train_index and args.test_index):
            if args.split_index:
                df_split = pd.read_csv(args.split_index)
                if not {"issue_key", "test"}.issubset(df_split.columns):
                    raise SystemExit(
                        "Combined split CSV must contain 'issue_key' and 'test' columns"
                    )
                df_split["test"] = df_split["test"].astype(int)
                train_keys = set(
                    df_split.loc[df_split["test"] == 0, "issue_key"].astype(str)
                )
                test_keys = set(
                    df_split.loc[df_split["test"] == 1, "issue_key"].astype(str)
                )
            else:
                df_tr = pd.read_csv(args.train_index)
                df_te = pd.read_csv(args.test_index)
                if "issue_key" not in df_tr.columns or "issue_key" not in df_te.columns:
                    raise SystemExit("Split CSVs must contain an 'issue_key' column")
                train_keys = set(df_tr["issue_key"].astype(str).tolist())
                test_keys = set(df_te["issue_key"].astype(str).tolist())
            key_series = df_feats["issue_key"].astype(str).reset_index(drop=True)
            train_mask = key_series.isin(train_keys).to_numpy()
            test_mask = key_series.isin(test_keys).to_numpy()

            # Baseline model with explicit split
            model = Pipeline(
                [
                    ("prep", preproc),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=300, random_state=args.split_seed, n_jobs=-1
                        ),
                    ),
                ]
            )
            model.fit(X[train_mask], y[train_mask])
            preds_a = model.predict(X[test_mask])
            y_te_a = y[test_mask]
        else:
            base_model, y_te_a, preds_a = train_baseline(
                X, y, preproc, seed=args.split_seed, test_size=args.test_size
            )

        if feedback_csv.exists():
            fb = pd.read_csv(feedback_csv)
            fb["feedback_type"] = fb["feedback_type"].str.lower()
            if args.split_index or (args.train_index and args.test_index):
                # Feedback-weighted with explicit split
                # Build like_rate weights aligned to X rows
                verdict = fb.groupby("issue_key")["feedback_type"].agg(
                    lambda s: (s == "like").mean()
                )
                weights_df = issues[["issue_key"]].merge(
                    verdict.rename("like_rate"),
                    left_on="issue_key",
                    right_index=True,
                    how="left",
                )
                w = weights_df["like_rate"].fillna(0.6).clip(0.2, 1.2).values
                model_fw = Pipeline(
                    [
                        ("prep", preproc),
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=300,
                                random_state=args.split_seed,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                model_fw.fit(
                    X[train_mask], y[train_mask], rf__sample_weight=w[train_mask]
                )
                preds_b = model_fw.predict(X[test_mask])
                y_te_b = y[test_mask]
            else:
                fw_model, y_te_b, preds_b = train_feedback_weighted(
                    X,
                    y,
                    preproc,
                    issues,
                    fb,
                    seed=args.split_seed,
                    test_size=args.test_size,
                )
            out = plots_dir / "comparison_baseline_vs_feedback.png"
            plot_comparison(y_te_a, preds_a, y_te_b, preds_b, out)
            print(
                f"Saved comparison plots to {out} and {out.with_name(out.stem + '_errors.png')}"
            )
        else:
            print("Feedback CSV not found; only baseline model trained.")
