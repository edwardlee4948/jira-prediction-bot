#!/usr/bin/env python3
"""
Transactional data analysis and prediction demo on synthetic Jira data.

- Loads synthetic issue-level data and event transitions
- Produces descriptive/diagnostic stats
- Engineers simple features and trains a baseline regressor to predict resolve_duration_hours
- Illustrates a human-in-the-loop feedback loop for incremental updates
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

try:
    # scikit-learn >= 1.6
    from sklearn.metrics import root_mean_squared_error
except Exception:  # pragma: no cover - fallback for older versions
    root_mean_squared_error = None  # type: ignore
    from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Visualization removed to keep this module focused on data and modeling


DATA_DIR = Path("experiments/jira_data")
ISSUES_CSV = DATA_DIR / "synthetic.csv"
EVENTS_CSV = DATA_DIR / "synthetic_events.csv"
FEEDBACK_CSV = DATA_DIR / "synthetic_feedback.csv"
MODEL_DIR = Path("experiments/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "resolve_duration_rf.joblib"


def load_data(
    issues_path: Path, events_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    issues = pd.read_csv(issues_path)
    events = pd.read_csv(events_path)

    # Basic type fixes
    for col in ["created_at", "first_response_at", "in_progress_at", "resolved_at"]:
        if col in issues.columns:
            issues[col] = pd.to_datetime(issues[col], errors="coerce")
    if "transitioned_at" in events.columns:
        events["transitioned_at"] = pd.to_datetime(
            events["transitioned_at"], errors="coerce"
        )

    return issues, events


def load_feedback(path: Path):
    if path.exists():
        fb = pd.read_csv(path)
        if "feedback_time" in fb.columns:
            fb["feedback_time"] = pd.to_datetime(fb["feedback_time"], errors="coerce")
        return fb
    return None


def descriptive_diagnostics(issues: pd.DataFrame, events: pd.DataFrame) -> None:
    print("=== Descriptive (Issues) ===")
    print("Rows:", len(issues))
    print(
        issues[
            ["issue_type", "priority", "environment", "resolve_duration_hours"]
        ].describe(include="all")
    )
    print("\nResolve duration by priority (median hours):")
    print(issues.groupby("priority")["resolve_duration_hours"].median().sort_index())
    print("\nResolve duration by issue type (median hours):")
    print(issues.groupby("issue_type")["resolve_duration_hours"].median().sort_values())
    print("\nSLA breach rate by priority:")
    if "sla_breached" in issues.columns:
        print(issues.groupby("priority")["sla_breached"].mean().round(3))

    print("\n=== Descriptive (Events) ===")
    print("Rows:", len(events))
    if {"from_status", "to_status"}.issubset(events.columns):
        print("Top transitions:")
        print(
            events.groupby(["from_status", "to_status"])["issue_key"]
            .count()
            .sort_values(ascending=False)
            .head(10)
        )
    if "minutes_in_from_status" in events.columns:
        print("\nMedian minutes per status:")
        print(
            events.groupby("from_status")["minutes_in_from_status"]
            .median()
            .sort_values(ascending=False)
            .head(10)
        )

    # Diagnostic: early signals
    if set(["event_type", "minutes_in_from_status"]).issubset(events.columns):
        early = events[events["event_index"] <= 2]
        early_time = (
            early.groupby("issue_key")["minutes_in_from_status"]
            .sum()
            .rename("early_minutes")
        )
        joined = issues.join(early_time, on="issue_key")
        corr = joined[["early_minutes", "resolve_duration_hours"]].corr().iloc[0, 1]
        print(f"\nDiagnostic: corr(early_minutes, resolve_duration_hours) = {corr:.3f}")


def build_features(issues: pd.DataFrame, events: pd.DataFrame):
    # Aggregate events per issue
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

    # Basic numeric fill
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

    # Target
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

    # Simple text features: title/context length
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

    # Keep key alignment for optional external splits
    key_series = df["issue_key"].astype(str).reset_index(drop=True)
    return X, y, preprocessor, key_series


def train_and_eval(
    X,
    y,
    preprocessor,
    test_size: float = 0.2,
    random_state: int = 42,
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
):
    if train_mask is not None and test_mask is not None:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_test, preds)
    else:
        # Fallback for older sklearn
        rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"Model: RandomForestRegressor | MAE={mae:.2f}h RMSE={rmse:.2f}h R2={r2:.3f}")
    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def save_model(model, path: Path):
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def incremental_update_with_feedback(model, X_new, y_new):
    # For demo: refit on feedback samples
    model.fit(X_new, y_new)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Analyze synthetic transactions and train predictor"
    )
    parser.add_argument("--issues", type=str, default=str(ISSUES_CSV))
    parser.add_argument("--events", type=str, default=str(EVENTS_CSV))
    parser.add_argument("--feedback", type=str, default=str(FEEDBACK_CSV))
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--simulate-feedback",
        action="store_true",
        help="Run a simple human-in-the-loop update demo",
    )
    # plots-dir removed
    parser.add_argument(
        "--split-index",
        type=str,
        help="Path to combined split CSV (columns: issue_key,test)",
    )
    args = parser.parse_args()

    issues, events = load_data(Path(args.issues), Path(args.events))
    feedback = load_feedback(Path(args.feedback))

    # 1) Descriptive/diagnostic/exploratory
    descriptive_diagnostics(issues, events)

    # 2) Feature engineering and model training
    X, y, preproc, key_series = build_features(issues, events)
    train_mask = test_mask = None
    if args.split_index:
        try:
            split_df = pd.read_csv(args.split_index)
            if not {"issue_key", "test"}.issubset(split_df.columns):
                raise ValueError("split_index must have columns: issue_key,test")
            split_df["test"] = split_df["test"].astype(int)
            train_keys = set(
                split_df.loc[split_df["test"] == 0, "issue_key"].astype(str)
            )
            test_keys = set(
                split_df.loc[split_df["test"] == 1, "issue_key"].astype(str)
            )
            train_mask = key_series.isin(train_keys).to_numpy()
            test_mask = key_series.isin(test_keys).to_numpy()
            n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
            print(f"Using external split_index: train={n_tr} test={n_te}")
            if n_tr == 0 or n_te == 0:
                print(
                    "Warning: one of the splits is empty after alignment; falling back to random split"
                )
                train_mask = test_mask = None
        except Exception as e:
            print(f"Failed to use split_index ({e}); falling back to random split")
            train_mask = test_mask = None

    model, metrics = train_and_eval(
        X,
        y,
        preproc,
        test_size=0.2,
        random_state=42,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    # Optional: summarize like/dislike feedback
    if feedback is not None and {"issue_key", "feedback_type"}.issubset(
        feedback.columns
    ):
        joined_fb = feedback.merge(issues[["issue_key"]], on="issue_key", how="inner")
        likes = (joined_fb["feedback_type"].str.lower() == "like").mean()
        print(
            f"Feedback summary over {len(joined_fb)} items: like={likes:.2%}, dislike={(1 - likes):.2%}"
        )
        # Visualization removed

    if args.save_model:
        save_model(model, MODEL_PATH)

    # 3) Human-in-the-loop feedback simulation
    if args.simulate_feedback:
        sample = issues.sample(n=min(100, len(issues)), random_state=123)
        sub_events = events[events["issue_key"].isin(sample["issue_key"])].copy()
        X_fb, y_fb, _, _ = build_features(sample, sub_events)
        rng = np.random.default_rng(123)
        y_fb_adj = y_fb * (1.0 + rng.normal(0, 0.05, size=len(y_fb)))
        model = incremental_update_with_feedback(model, X_fb, y_fb_adj)
        preds = model.predict(X_fb)
        mae_fb = mean_absolute_error(y_fb_adj, preds)
        print(
            f"After feedback update on {len(sample)} samples: MAE={mae_fb:.2f}h (on feedback set)"
        )


if __name__ == "__main__":
    main()
