#!/usr/bin/env python3
"""
Generate synthetic Jira-like issue data for ML experiments.

Outputs a CSV with columns commonly used to predict time to resolve.
Includes short text fields (title, context) plus structured features and timestamps.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any

random.seed(42)

# --- Configurable vocabularies ---
PROJECTS = ["APP", "PLAT", "PAY", "DATA", "OPS"]
ISSUE_TYPES = ["Bug", "Task", "Story"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
SEVERITIES = ["Blocker", "Critical", "Major", "Minor", "Trivial"]
CUSTOMER_IMPACT = ["High", "Medium", "Low", "None"]
ENVIRONMENTS = ["Prod", "Staging", "Dev", "QA"]
COMPONENTS = [
    "API Gateway",
    "Web UI",
    "Mobile",
    "Auth",
    "Billing",
    "Data Pipeline",
    "Notifications",
    "Search",
]
LABELS = [
    "regression",
    "oncall",
    "hotfix",
    "customer",
    "backend",
    "frontend",
    "infra",
    "performance",
]


TITLE_TEMPLATES = [
    "{component} {issue_type_lower} causes {impact_lower} impact in {env}",
    "{component} {issue_type} when {trigger}",
    "Improve {component} {metric} in {env}",
    "{component} returns {error_code} on {action}",
    "Add support for {feature} in {component}",
]

TRIGGERS = [
    "user logs in",
    "payment is processed",
    "feature flag is toggled",
    "cache is cold",
    "deploy happens",
]

METRICS = ["latency", "throughput", "error rate", "CPU usage", "memory usage"]
ERROR_CODES = ["400", "401", "403", "404", "409", "429", "500", "502", "503"]
FEATURES = ["OAuth2", "webhooks", "CSV export", "SAML SSO", "dark mode", "retry"]
ACTIONS = [
    "POST /v1/login",
    "GET /search",
    "PUT /users/{id}",
    "checkout flow",
    "upload",
]


DESC_SENTENCES = [
    "Steps to reproduce: {steps}.",
    "Observed behavior: {observed}.",
    "Expected behavior: {expected}.",
    "Logs indicate {logs}.",
    "Happens intermittently under {condition}.",
    "Customer {account} reported the issue via {channel}.",
    "Workaround: {workaround}.",
]

STEPS = [
    "navigate to the dashboard and click on settings",
    "submit the form with valid credentials",
    "retry the request three times",
    "clear cookies then refresh the page",
    "switch networks between wifi and cellular",
]
OBSERVED = [
    "request times out after 30 seconds",
    "500 Internal Server Error is returned",
    "UI freezes and becomes unresponsive",
    "data is not persisted to the database",
    "user remains logged out",
]
EXPECTED = [
    "request completes within SLA",
    "200 OK response with JSON payload",
    "UI remains responsive",
    "data is saved and retrievable",
    "user is authenticated and redirected",
]
LOG_HINTS = [
    "null pointer in service layer",
    "timeout connecting to postgres",
    "rate limit exceeded",
    "missing feature flag",
    "JWT signature validation failed",
]
CONDITIONS = [
    "high load",
    "low latency network",
    "peak traffic",
    "degraded DB",
    "deploy",
]
ACCOUNTS = ["Acme Corp", "Globex", "Initech", "Umbrella", "Soylent"]
CHANNELS = ["email", "Slack", "support portal", "phone call"]
WORKAROUNDS = [
    "retry after 60 seconds",
    "toggle the feature flag off and on",
    "use legacy endpoint",
    "clear cache and refresh",
    "rollback to previous version",
]


def rand_choice(seq):
    return random.choice(seq)


def random_title(issue_type: str, component: str, env: str) -> str:
    template = rand_choice(TITLE_TEMPLATES)
    return (
        template.replace("{component}", component)
        .replace("{issue_type}", issue_type)
        .replace("{issue_type_lower}", issue_type.lower())
        .replace("{env}", env)
        .replace("{impact_lower}", rand_choice(CUSTOMER_IMPACT).lower())
        .replace("{trigger}", rand_choice(TRIGGERS))
        .replace("{metric}", rand_choice(METRICS))
        .replace("{error_code}", rand_choice(ERROR_CODES))
        .replace("{feature}", rand_choice(FEATURES))
        .replace("{action}", rand_choice(ACTIONS))
    )


def random_description() -> str:
    parts = []
    for sentence in DESC_SENTENCES:
        s = (
            sentence.replace("{steps}", rand_choice(STEPS))
            .replace("{observed}", rand_choice(OBSERVED))
            .replace("{expected}", rand_choice(EXPECTED))
            .replace("{logs}", rand_choice(LOG_HINTS))
            .replace("{condition}", rand_choice(CONDITIONS))
            .replace("{account}", rand_choice(ACCOUNTS))
            .replace("{channel}", rand_choice(CHANNELS))
            .replace("{workaround}", rand_choice(WORKAROUNDS))
        )
        parts.append(s)
    # Randomly add a pseudo stack trace or code block for realism
    if random.random() < 0.4:
        parts.append(
            "Stack trace: java.lang.NullPointerException at com.example.Service.handle(Service.java:42)"
        )
    return "\n".join(parts)


def random_labels(issue_type: str) -> List[str]:
    labels = set()
    if issue_type == "Bug":
        labels.update(["bug", "triage"])
        if random.random() < 0.3:
            labels.add("regression")
        if random.random() < 0.2:
            labels.add("hotfix")
    else:
        labels.add("feature" if issue_type == "Story" else "maintenance")
    # Add 0-2 random extra labels
    for _ in range(random.randint(0, 2)):
        labels.add(rand_choice(LABELS))
    return sorted(labels)


def weighted_resolution_hours(
    issue_type: str,
    priority: str,
    severity: str | None,
    description_len: int,
    comments: int,
    reopened: int,
    experience_years: int,
    env: str,
    customer_impact: str,
) -> float:
    # Base hours by type
    base = {
        "Bug": random.uniform(8, 72),
        "Task": random.uniform(12, 120),
        "Story": random.uniform(16, 160),
    }[issue_type]

    # Priority effect
    base *= {"P1": 0.6, "P2": 0.85, "P3": 1.0, "P4": 1.3}[priority]

    # Severity effect for bugs
    if issue_type == "Bug" and severity:
        base *= {
            "Blocker": 0.5,
            "Critical": 0.7,
            "Major": 1.0,
            "Minor": 1.2,
            "Trivial": 1.3,
        }[severity]

    # Customer impact
    base *= {"High": 0.7, "Medium": 0.9, "Low": 1.0, "None": 1.1}[customer_impact]

    # Environment (Prod gets attention faster, QA/Dev slower)
    base *= {"Prod": 0.8, "Staging": 0.95, "QA": 1.05, "Dev": 1.1}.get(env, 1.0)

    # More comments often means a trickier issue
    base *= 1.0 + min(comments, 20) * 0.02

    # Reopened issues take longer
    base *= 1.0 + reopened * 0.25

    # Longer descriptions correlate with complexity, but saturate
    base *= 1.0 + min(description_len / 800.0, 0.3)

    # Experience reduces time
    base *= max(0.6, 1.0 - (experience_years * 0.03))

    # Noise
    noise = random.normalvariate(1.0, 0.15)
    base *= max(0.3, noise)

    # Clamp to reasonable bounds
    return float(max(1.0, min(base, 24 * 30)))  # up to ~30 days


def pseudo_email(name: str) -> str:
    domain = rand_choice(["example.com", "corp.local", "company.io", "startup.dev"])
    handle = name.lower().replace(" ", ".")
    return f"{handle}@{domain}"


def random_name() -> str:
    first = rand_choice(
        [
            "Alex",
            "Sam",
            "Jordan",
            "Taylor",
            "Casey",
            "Riley",
            "Morgan",
            "Jamie",
            "Avery",
        ]
    )  # gender-neutral
    last = rand_choice(
        [
            "Lee",
            "Kim",
            "Garcia",
            "Patel",
            "Nguyen",
            "Smith",
            "Brown",
            "Davis",
            "Martinez",
        ]
    )
    return f"{first} {last}"


@dataclass
class Issue:
    # Identifiers
    project_key: str
    issue_key: str

    # Textual features
    title: str  # aka summary
    context: str  # aka description

    # Categorical/structured features
    issue_type: str
    priority: str
    severity: str | None
    environment: str
    affected_component: str
    labels: str  # comma-separated
    story_points: int | None

    # People
    reporter: str
    reporter_email: str
    assignee: str
    assignee_email: str
    assignee_experience_years: int
    reporter_team: str
    assignee_team: str

    # Activity counts
    num_comments: int
    num_watchers: int
    num_attachments: int
    num_linked_issues: int
    reopened_count: int
    pull_requests_linked: int

    # Timeline
    created_at: str
    first_response_at: str
    in_progress_at: str
    resolved_at: str

    # Targets/derived
    time_to_first_response_hours: float
    time_to_in_progress_hours: float
    resolve_duration_hours: float  # target
    sla_breached: int
    customer_impact: str
    ci_cd_status: str


def generate_issue(idx: int, now: datetime) -> Issue:
    project = rand_choice(PROJECTS)
    issue_type = rand_choice(ISSUE_TYPES)
    priority = random.choices(PRIORITIES, weights=[0.1, 0.3, 0.4, 0.2])[0]
    severity = rand_choice(SEVERITIES) if issue_type == "Bug" else None
    env = random.choices(ENVIRONMENTS, weights=[0.5, 0.2, 0.2, 0.1])[0]
    component = rand_choice(COMPONENTS)
    labels = random_labels(issue_type)

    title = random_title(issue_type, component, env)
    desc = random_description()

    story_points = None
    if issue_type in ("Story", "Task"):
        story_points = random.choices(
            [1, 2, 3, 5, 8, 13], weights=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]
        )[0]

    reporter = random_name()
    assignee = random_name()
    experience_years = random.randint(0, 15)
    teams = [
        "Core",
        "Payments",
        "Platform",
        "Data",
        "Ops",
        "SRE",
        "Frontend",
        "Backend",
    ]
    reporter_team = rand_choice(teams)
    assignee_team = rand_choice(teams)

    num_comments = max(0, int(random.normalvariate(3 if issue_type == "Bug" else 2, 2)))
    num_watchers = max(0, int(random.normalvariate(2, 2)))
    num_attachments = random.randint(0, 5)
    num_linked_issues = random.randint(0, 3)
    reopened_count = random.choices([0, 1, 2, 3], weights=[0.8, 0.15, 0.04, 0.01])[0]
    pull_requests_linked = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]

    customer_impact = random.choices(CUSTOMER_IMPACT, weights=[0.15, 0.35, 0.35, 0.15])[
        0
    ]
    ci_cd_status = (
        rand_choice(["pass", "fail", "na"]) if pull_requests_linked > 0 else "na"
    )

    description_len = len(desc)
    resolve_hours = weighted_resolution_hours(
        issue_type,
        priority,
        severity,
        description_len,
        num_comments,
        reopened_count,
        experience_years,
        env,
        customer_impact,
    )

    # Build timeline
    created = now - timedelta(
        days=random.randint(0, 120),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    t_first_resp = (
        random.uniform(0.2, 12.0)
        if priority in ("P1", "P2")
        else random.uniform(0.5, 36.0)
    )
    t_in_progress = t_first_resp + random.uniform(0.1, 24.0)
    first_response_at = created + timedelta(hours=t_first_resp)
    in_progress_at = created + timedelta(hours=t_in_progress)
    resolved_at = created + timedelta(hours=resolve_hours)

    # SLA breach if P1/P2 and resolve > 48/72h (simple heuristic)
    sla_threshold = 48.0 if priority == "P1" else 72.0 if priority == "P2" else 120.0
    sla_breached = int(resolve_hours > sla_threshold)

    return Issue(
        project_key=project,
        issue_key=f"{project}-{idx}",
        title=title,
        context=desc,
        issue_type=issue_type,
        priority=priority,
        severity=severity,
        environment=env,
        affected_component=component,
        labels=",".join(labels),
        story_points=story_points,
        reporter=reporter,
        reporter_email=pseudo_email(reporter),
        assignee=assignee,
        assignee_email=pseudo_email(assignee),
        assignee_experience_years=experience_years,
        reporter_team=reporter_team,
        assignee_team=assignee_team,
        num_comments=num_comments,
        num_watchers=num_watchers,
        num_attachments=num_attachments,
        num_linked_issues=num_linked_issues,
        reopened_count=reopened_count,
        pull_requests_linked=pull_requests_linked,
        created_at=created.isoformat(timespec="seconds"),
        first_response_at=first_response_at.isoformat(timespec="seconds"),
        in_progress_at=in_progress_at.isoformat(timespec="seconds"),
        resolved_at=resolved_at.isoformat(timespec="seconds"),
        time_to_first_response_hours=round(t_first_resp, 2),
        time_to_in_progress_hours=round(t_in_progress, 2),
        resolve_duration_hours=round(resolve_hours, 2),
        sla_breached=sla_breached,
        customer_impact=customer_impact,
        ci_cd_status=ci_cd_status,
    )


def write_csv(issues: List[Issue], path: str) -> None:
    if not issues:
        return
    fieldnames = list(asdict(issues[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            row: Dict[str, Any] = asdict(issue)
            writer.writerow(row)


# --- Transitions/events generation ---
STATUSES = [
    "Open",
    "Triaged",
    "In Progress",
    "In Review",
    "QA",
    "Done",
    "Reopened",
]


def choose_status_path(issue_type: str) -> List[str]:
    # Basic workflow with optional steps
    if issue_type == "Bug":
        path = ["Open", "Triaged", "In Progress"]
        if random.random() < 0.75:
            path.append("In Review")
        if random.random() < 0.6:
            path.append("QA")
        path.append("Done")
    else:
        path = ["Open", "In Progress"]
        if random.random() < 0.65:
            path.append("In Review")
        if random.random() < 0.45:
            path.append("QA")
        path.append("Done")
    return path


def allocate_state_durations(
    issue_type: str, reopened_count: int, total_hours: float
) -> List[tuple[str, float]]:
    path = choose_status_path(issue_type)
    # Base weights per status
    weights = []
    for s in path:
        if s == "Open":
            w = random.uniform(0.05, 0.2)
        elif s == "Triaged":
            w = random.uniform(0.05, 0.15)
        elif s == "In Progress":
            w = random.uniform(0.4, 0.6)
        elif s == "In Review":
            w = random.uniform(0.05, 0.15)
        elif s == "QA":
            w = random.uniform(0.05, 0.15)
        elif s == "Done":
            w = 0.0  # terminal state; no duration spent
        else:
            w = 0.0
        weights.append(w)

    # Reopen cycles increase work-phase weights
    if reopened_count > 0:
        inc = reopened_count * random.uniform(0.08, 0.25)
        for i, s in enumerate(path):
            if s in ("In Progress", "In Review", "QA"):
                weights[i] *= 1.0 + inc

    # Normalize non-terminal weights
    non_terminal_indices = [i for i, s in enumerate(path) if s != "Done"]
    total_w = sum(weights[i] for i in non_terminal_indices)
    if total_w <= 0:
        total_w = 1.0
    for i in non_terminal_indices:
        weights[i] /= total_w

    durations = []
    for s, w in zip(path, weights):
        hours = 0.0 if s == "Done" else (total_hours * w)
        durations.append((s, hours))
    return durations


def build_transitions(issue: Issue) -> List[Dict[str, Any]]:
    # Allocate durations so they sum to resolve_duration_hours
    durations = allocate_state_durations(
        issue.issue_type, issue.reopened_count, issue.resolve_duration_hours
    )
    # Construct timeline starting at created_at
    t = datetime.fromisoformat(issue.created_at)
    events: List[Dict[str, Any]] = []
    current_status = durations[0][0]

    # Ensure first status is Open at created_at
    if current_status != "Open":
        current_status = "Open"
        durations.insert(0, ("Open", max(0.25, random.uniform(0.25, 8.0))))

    # Optionally insert a First Response marker before moving to next status
    first_resp_time = datetime.fromisoformat(issue.first_response_at)
    if first_resp_time > t:
        events.append(
            {
                "issue_key": issue.issue_key,
                "event_index": 0,
                "event_type": "first_response",
                "from_status": current_status,
                "to_status": current_status,
                "actor": issue.assignee,
                "actor_email": issue.assignee_email,
                "transitioned_at": first_resp_time.isoformat(timespec="seconds"),
                "minutes_in_from_status": round(
                    (first_resp_time - t).total_seconds() / 60.0, 2
                ),
                "comments_delta": 0,
                "prs_delta": 0,
            }
        )

    idx = 1 if events else 0
    for i in range(len(durations) - 1):
        status, hours = durations[i]
        next_status, _ = durations[i + 1]
        # time spent in this status
        delta = timedelta(hours=hours)
        t_next = t + delta
        actor = issue.assignee
        # Occasionally use reporter for early transitions or reopen
        if status in ("Open", "Triaged") and random.random() < 0.3:
            actor = issue.reporter
        if next_status == "Reopened" and random.random() < 0.5:
            actor = issue.reporter
        events.append(
            {
                "issue_key": issue.issue_key,
                "event_index": idx,
                "event_type": "status_change",
                "from_status": status,
                "to_status": next_status,
                "actor": actor,
                "actor_email": issue.reporter_email
                if actor == issue.reporter
                else issue.assignee_email,
                "transitioned_at": t_next.isoformat(timespec="seconds"),
                "minutes_in_from_status": round(delta.total_seconds() / 60.0, 2),
                "comments_delta": max(0, int(random.normalvariate(0.6, 1.0))),
                "prs_delta": 1
                if (
                    status == "In Review"
                    and issue.pull_requests_linked > 0
                    and random.random() < 0.6
                )
                else 0,
            }
        )
        t = t_next
        idx += 1

    # Align final timestamp with resolved_at if needed
    resolved_at = datetime.fromisoformat(issue.resolved_at)
    if t < resolved_at:
        events.append(
            {
                "issue_key": issue.issue_key,
                "event_index": idx,
                "event_type": "status_change",
                "from_status": durations[-2][0]
                if len(durations) >= 2
                else "In Progress",
                "to_status": "Done",
                "actor": issue.assignee,
                "actor_email": issue.assignee_email,
                "transitioned_at": resolved_at.isoformat(timespec="seconds"),
                "minutes_in_from_status": round(
                    (resolved_at - t).total_seconds() / 60.0, 2
                ),
                "comments_delta": max(0, int(random.normalvariate(0.2, 0.5))),
                "prs_delta": 0,
            }
        )

    return events


def write_events_csv(events: List[Dict[str, Any]], path: str) -> None:
    if not events:
        return
    fieldnames = [
        "issue_key",
        "event_index",
        "event_type",
        "from_status",
        "to_status",
        "actor",
        "actor_email",
        "transitioned_at",
        "minutes_in_from_status",
        "comments_delta",
        "prs_delta",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            writer.writerow(e)


# --- Synthetic feedback generation ---
def generate_feedback(issues: List[Issue], ratio: float = 0.2) -> List[Dict[str, Any]]:
    """Generate sparse, biased human feedback (like/dislike) about the prediction.

    Selection bias: More likely to get feedback for P1/P4, Bugs, Prod, High impact,
    SLA-breached, and customer/oncall/hotfix-labeled issues. Less likely in Dev/QA.

    Verdict bias: Harder/longer issues (long duration, many comments, reopened, high
    impact, SLA-breached) receive more dislikes.
    """
    if ratio <= 0:
        return []

    def selection_multiplier(iss: Issue) -> float:
        m = 1.0
        # Priority
        if iss.priority == "P1":
            m *= 1.8
        elif iss.priority == "P4":
            m *= 1.3
        else:
            m *= {"P2": 1.2, "P3": 1.0}.get(iss.priority, 1.0)
        # Type
        if iss.issue_type == "Bug":
            m *= 1.35
        # Environment
        env_w = {"Prod": 1.5, "Staging": 1.1, "QA": 0.85, "Dev": 0.75}
        m *= env_w.get(iss.environment, 1.0)
        # Customer impact
        impact_w = {"High": 1.6, "Medium": 1.2, "Low": 0.95, "None": 0.8}
        m *= impact_w.get(iss.customer_impact, 1.0)
        # SLA
        if getattr(iss, "sla_breached", 0):
            m *= 1.7
        # Labels that often attract attention
        labels = (iss.labels or "").lower()
        for tag, w in [("customer", 1.25), ("oncall", 1.2), ("hotfix", 1.2)]:
            if tag in labels:
                m *= w
        return m

    def dislike_probability(iss: Issue) -> float:
        # Difficulty proxy
        dur = float(getattr(iss, "resolve_duration_hours", 0.0))
        comments = int(getattr(iss, "num_comments", 0))
        reopened = int(getattr(iss, "reopened_count", 0))
        sp = getattr(iss, "story_points", 0) or 0
        base = 0.15  # baseline dislike rate
        # Normalize and accumulate difficulty
        d = 0.0
        d += min(dur / 120.0, 1.5)  # up to ~5 days strongly increases difficulty
        d += min(comments / 12.0, 1.0) * 0.5
        d += min(reopened, 3) * 0.3
        d += 0.2 if sp and sp >= 8 else 0.0
        if getattr(iss, "sla_breached", 0):
            d += 0.5
        if iss.customer_impact == "High":
            d += 0.3
        # Map difficulty to dislike probability
        scale = max(0.0, min(d / 2.5, 1.0))
        p_dislike = base + 0.7 * scale  # 0.15 .. 0.85
        return max(0.05, min(p_dislike, 0.9))

    rows: List[Dict[str, Any]] = []
    for iss in issues:
        p_select = min(0.95, ratio * selection_multiplier(iss))
        if random.random() >= p_select:
            continue
        # Feedback time shortly after first response (or sometimes later)
        base_time = datetime.fromisoformat(iss.first_response_at)
        jitter_hours = (
            random.uniform(0.1, 12.0)
            if random.random() < 0.8
            else random.uniform(12.0, 72.0)
        )
        fb_time = base_time + timedelta(hours=jitter_hours)

        verdict = "dislike" if random.random() < dislike_probability(iss) else "like"

        rows.append(
            {
                "issue_key": iss.issue_key,
                "feedback_time": fb_time.isoformat(timespec="seconds"),
                "feedback_type": verdict,
            }
        )

    # Ensure we return at least one row if ratio > 0
    if not rows and issues:
        iss = random.choice(issues)
        base_time = datetime.fromisoformat(iss.first_response_at)
        fb_time = base_time + timedelta(hours=random.uniform(0.1, 12.0))
        verdict = "dislike" if random.random() < dislike_probability(iss) else "like"
        rows.append(
            {
                "issue_key": iss.issue_key,
                "feedback_time": fb_time.isoformat(timespec="seconds"),
                "feedback_type": verdict,
            }
        )

    return rows


def write_feedback_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = ["issue_key", "feedback_time", "feedback_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Jira data")
    parser.add_argument(
        "--rows", type=int, default=1000, help="Number of issues to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/jira_data/synthetic.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--events-output",
        type=str,
        default="experiments/jira_data/synthetic_events.csv",
        help="Path to output events CSV file",
    )
    parser.add_argument(
        "--feedback-output",
        type=str,
        default="experiments/jira_data/synthetic_feedback.csv",
        help="Path to output feedback CSV file",
    )
    parser.add_argument(
        "--feedback-ratio",
        type=float,
        default=0.2,
        help="Fraction of issues receiving feedback",
    )
    # Optional: write reproducible combined split index CSV
    parser.add_argument(
        "--write-split",
        action="store_true",
        help="Write combined split_index.csv (issue_key,test) for reuse in visualization",
    )
    parser.add_argument(
        "--split-test-size",
        type=float,
        default=0.2,
        help="Test size fraction for the split (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the split (default: 42)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="experiments/jira_data/splits",
        help="Directory to write split CSV",
    )
    args = parser.parse_args()

    now = datetime.utcnow()
    issues = [generate_issue(i + 1, now) for i in range(args.rows)]
    write_csv(issues, args.output)
    # Build events after issues so they align with timestamps
    all_events: List[Dict[str, Any]] = []
    for iss in issues:
        all_events.extend(build_transitions(iss))
    write_events_csv(all_events, args.events_output)
    # Generate feedback
    fb_rows = generate_feedback(issues, ratio=args.feedback_ratio)
    write_feedback_csv(fb_rows, args.feedback_output)
    print(
        f"Wrote {len(issues)} rows to {args.output}, {len(all_events)} events to {args.events_output}, and {len(fb_rows)} feedback rows to {args.feedback_output}"
    )

    # Write split index CSV if requested
    if args.write_split:
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise SystemExit(
                f"Writing split requires pandas and scikit-learn in the environment: {e}"
            )
        # Build a DataFrame of keys to ensure stable ordering
        df_keys = pd.DataFrame(
            {
                "row_index": list(range(len(issues))),
                "issue_key": [iss.issue_key for iss in issues],
            }
        )
        idx = df_keys.index.to_numpy()
        idx_tr, idx_te = train_test_split(
            idx, test_size=args.split_test_size, random_state=args.split_seed
        )
        train_df = df_keys.loc[idx_tr].copy()
        test_df = df_keys.loc[idx_te].copy()
        # Write a combined, minimal split file for convenience: issue_key,test
        combined = pd.concat(
            [
                pd.DataFrame(
                    {
                        "issue_key": train_df["issue_key"].astype(str).values,
                        "test": 0,
                    }
                ),
                pd.DataFrame(
                    {
                        "issue_key": test_df["issue_key"].astype(str).values,
                        "test": 1,
                    }
                ),
            ],
            ignore_index=True,
        )
        from pathlib import Path

        split_dir = Path(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        combined_path = split_dir / "split_index.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Wrote split CSV: {combined_path}")


if __name__ == "__main__":
    main()
