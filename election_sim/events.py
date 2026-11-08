"""Deterministic audience and signal preparation for scheduled news/events."""

import hashlib

import numpy as np
import pandas as pd

from .campaign.effects import GROUP_MATCHERS
from .support import ISSUES


def _present(value) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def prepare_events(events: pd.DataFrame, voters: pd.DataFrame, candidate_ids: list[str],
                   seed: list[int], days: int, constituency_id: int,
                   default_memory_half_life_days: float = 7.0,
                   default_reminder_probability: float = 0.0) -> list[dict]:
    """Validate and compile constituency-scoped event rows for daily application.

    Event duration is the number of active days. Each event's total scheduled
    signal is spread evenly over those days. The remaining effect then decays
    with an optional half-life; a zero half-life means permanent memory.
    """
    if events is None or events.empty:
        return []
    required = {"event_id", "event_day", "duration", "intensity", "credibility",
                "target_group", "issue", "candidate_id", "candidate_effect"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    if "event_sensitivity" not in voters:
        raise ValueError("voters need event_sensitivity when events are supplied")
    sensitivity = pd.to_numeric(voters.event_sensitivity, errors="coerce").to_numpy(float)
    if not np.isfinite(sensitivity).all() or ((sensitivity < 0) | (sensitivity > 1)).any():
        raise ValueError("event_sensitivity must be finite and in [0, 1]")

    # The normal loader already returns this scope slice. Filter defensively
    # too, because callers may pass the complete events table directly.
    scoped_events = events
    if {"constituency_id", "region_id"}.issubset(events.columns):
        region_id = voters.region_id.iloc[0] if "region_id" in voters else None
        cons_values = pd.to_numeric(events.constituency_id, errors="coerce")
        region_values = pd.to_numeric(events.region_id, errors="coerce")
        local = cons_values.eq(constituency_id)
        regional = cons_values.isna() & region_values.eq(region_id) if region_id is not None else False
        state = cons_values.isna() & region_values.isna()
        scoped_events = events.loc[local | regional | state]

    issue_columns = [f"issue_{issue}" for issue in ISSUES]
    if all(column in voters for column in issue_columns):
        importance = voters[issue_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(importance).all() or (importance < 0).any():
            raise ValueError("voter issue importances must be finite and nonnegative")
        totals = importance.sum(axis=1)
        importance = np.divide(importance, totals[:, None], out=np.zeros_like(importance),
                               where=totals[:, None] > 0)
    else:
        importance = None

    compiled = []
    seen = set()
    for row in scoped_events.sort_values(["event_day", "event_id"], kind="stable").to_dict("records"):
        event_id = str(row["event_id"])
        if not event_id or event_id in seen:
            raise ValueError("event_id values must be present and unique")
        seen.add(event_id)
        start, duration = row["event_day"], row["duration"]
        if (isinstance(start, bool) or not float(start).is_integer() or not 1 <= int(start) <= days
                or isinstance(duration, bool) or not float(duration).is_integer() or int(duration) < 1):
            raise ValueError(f"event {event_id} has invalid event_day or duration")
        duration = int(duration)
        intensity, credibility = float(row["intensity"]), float(row["credibility"])
        if not np.isfinite([intensity, credibility]).all() or not (0 <= intensity <= 1 and 0 <= credibility <= 1):
            raise ValueError(f"event {event_id} intensity and credibility must be in [0, 1]")
        group = row["target_group"]
        if group not in GROUP_MATCHERS:
            raise ValueError(f"event {event_id} has unknown target_group {group!r}")
        try:
            audience = np.flatnonzero(GROUP_MATCHERS[group](voters))
        except AttributeError as exc:
            raise ValueError(f"voters lack attributes needed for event target_group {group!r}") from exc
        issue = row["issue"]
        if issue == "general":
            issue_index = None
            issue_weight = np.ones(len(voters))
        elif issue in ISSUES:
            column_index = ISSUES.index(issue)
            issue_index = column_index
            issue_weight = importance[:, column_index] if importance is not None else np.ones(len(voters))
        else:
            raise ValueError(f"event {event_id} has unknown issue {issue!r}")
        weights = sensitivity * issue_weight
        weights[np.setdiff1d(np.arange(len(voters)), audience, assume_unique=True)] = 0.0
        candidate_id = row["candidate_id"]
        associated_candidate_id = str(candidate_id) if _present(candidate_id) else None
        salt = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8], 16)
        shock_mode = row.get("shock_mode")
        shock_mode = str(shock_mode) if _present(shock_mode) else "balanced"
        if shock_mode not in {"balanced", "directional"}:
            raise ValueError(f"event {event_id} has unknown shock_mode {shock_mode!r}")
        opinion_direction = row.get("opinion_direction")
        opinion_direction = (None if not _present(opinion_direction)
                             else float(opinion_direction))
        issue_direction = row.get("issue_direction")
        issue_direction = (None if not _present(issue_direction)
                           else float(issue_direction))
        for field, value in (("opinion_direction", opinion_direction),
                             ("issue_direction", issue_direction)):
            if value is not None and (not np.isfinite(value) or not -1 <= value <= 1):
                raise ValueError(f"event {event_id} {field} must be in [-1, 1]")
        if issue_direction is not None and issue_index is None:
            raise ValueError(f"event {event_id} issue_direction requires a specific issue")
        if issue_direction is not None and shock_mode != "directional":
            raise ValueError(f"event {event_id} issue_direction requires shock_mode='directional'")
        if (issue_index is not None and issue_direction is not None
                and opinion_direction is not None
                and not np.isclose(issue_direction, opinion_direction)):
            raise ValueError(f"event {event_id} issue_direction and opinion_direction conflict")
        if associated_candidate_id is not None and associated_candidate_id in candidate_ids:
            if (shock_mode != "balanced" or opinion_direction is not None
                    or issue_direction is not None):
                raise ValueError(f"event {event_id} candidate signals cannot set a neutral shock direction")
            candidate_id = associated_candidate_id
            effect = float(row["candidate_effect"])
            if not np.isfinite(effect) or not -1 <= effect <= 1:
                raise ValueError(f"event {event_id} candidate_effect must be in [-1, 1]")
            direction = None
        else:
            # A region event can name a candidate from another constituency.
            # That person is not on this ballot, so don't apply their result
            # directly to any local candidate; treat the issue signal as
            # neutral for this constituency.
            candidate_id, effect = None, None
            direction = np.zeros(len(voters), dtype=float)
            if shock_mode == "directional":
                configured_direction = (issue_direction if issue_direction is not None
                                        else opinion_direction)
                if configured_direction is None:
                    field = "issue_direction" if issue_index is not None else "opinion_direction"
                    raise ValueError(f"event {event_id} directional shock needs {field}")
                direction[audience] = configured_direction
            elif len(audience) > 1:
                # A seeded zero-mean audience direction avoids silently
                # assigning political meaning to balanced neutral news.
                event_rng = np.random.default_rng([*seed, salt])
                values = event_rng.normal(size=len(audience))
                audience_weights = weights[audience]
                if audience_weights.sum() > 0:
                    values -= np.average(values, weights=audience_weights)
                else:
                    values[:] = 0.0
                max_abs = np.abs(values).max()
                if max_abs:
                    values /= max_abs
                direction[audience] = values
        def optional_number(name, default):
            value = row.get(name)
            return default if not _present(value) else float(value)

        occurrence_probability = optional_number("occurrence_probability", 1.0)
        condition_probability = optional_number("condition_probability", 1.0)
        reminder_probability = optional_number("reminder_probability", default_reminder_probability)
        memory_half_life = optional_number("memory_half_life_days", default_memory_half_life_days)
        for name, value in (("occurrence_probability", occurrence_probability),
                            ("condition_probability", condition_probability),
                            ("reminder_probability", reminder_probability)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"event {event_id} {name} must be in [0, 1]")
        if not np.isfinite(memory_half_life) or memory_half_life < 0:
            raise ValueError(f"event {event_id} memory_half_life_days must be nonnegative")
        dependency = row.get("depends_on_event_id")
        dependency = str(dependency) if _present(dependency) else None
        structural_effect = row.get("structural_effect")
        structural_effect = str(structural_effect) if _present(structural_effect) else "none"
        if structural_effect not in {"none", "candidate_withdrawal"}:
            raise ValueError(f"event {event_id} has unknown structural_effect {structural_effect!r}")
        if structural_effect == "candidate_withdrawal" and candidate_id is None:
            raise ValueError(f"event {event_id} candidate withdrawal requires a local candidate_id")
        reminder_rng = np.random.default_rng([*seed, salt, 2])
        compiled.append({"event_id": event_id, "start_day": int(start), "duration": duration,
                         "intensity": intensity, "credibility": credibility,
                         "audience": audience, "weights": weights, "issue": issue,
                         "issue_index": issue_index, "issue_direction": issue_direction,
                         "candidate_id": candidate_id,
                         "associated_candidate_id": associated_candidate_id,
                         "source_type": row.get("source_type"),
                         "event_type": (str(row["event_type"]) if _present(row.get("event_type"))
                                        else None),
                         "candidate_effect": effect, "direction": direction,
                         "shock_mode": shock_mode,
                         "memory_half_life_days": memory_half_life,
                         "reminder_probability": reminder_probability,
                         "occurrence_probability": occurrence_probability,
                         "condition_probability": condition_probability,
                         "depends_on_event_id": dependency,
                         "structural_effect": structural_effect,
                         "occurs": False, "contribution": np.zeros(len(voters)),
                         "reminder_rng": reminder_rng})
    by_id = {event["event_id"]: event for event in compiled}
    for event in compiled:
        dependency = event["depends_on_event_id"]
        if dependency is not None:
            if dependency not in by_id:
                raise ValueError(f"event {event['event_id']} depends on unknown or out-of-scope event {dependency}")
            parent = by_id[dependency]
            if parent["start_day"] >= event["start_day"]:
                raise ValueError(f"event {event['event_id']} dependency must start earlier")
        # Occurrence draws are per-event, so changing input row order or adding
        # another event cannot shift this event's outcome.
        salt = int(hashlib.sha256(event["event_id"].encode("utf-8")).hexdigest()[:8], 16)
        occurrence_rng = np.random.default_rng([*seed, salt, 1])
        parent_occurs = dependency is None or by_id[dependency]["occurs"]
        probability = event["occurrence_probability"]
        if dependency is not None:
            probability *= event["condition_probability"]
        event["occurs"] = parent_occurs and occurrence_rng.random() < probability
    return compiled


def active_on(event: dict, day: int) -> bool:
    return (event["occurs"] and event["start_day"] <= day
            < event["start_day"] + event["duration"])

