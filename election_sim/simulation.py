from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .campaign.effects import (GROUP_MATCHERS, action_rng, apply_ordinary, apply_smear,
                               assign_communities,
                               budget, outgoing_influence_scores, select_recipients)
from .campaign.models import CampaignAction, check_actions
from .dynamics import social_step
from .events import active_on as event_active_on, prepare_events
from .network import build_incoming
from .support import candidate_support
from .voting import election_result


@dataclass
class RunResult:
    daily_summary: pd.DataFrame
    daily_support: pd.DataFrame
    opinion_distribution: pd.DataFrame
    candidate_results: pd.DataFrame
    final_state: pd.DataFrame
    sampled_state: pd.DataFrame
    turnout: dict
    diagnostics: dict
    metadata: dict

    def save(self, output_dir: str | Path) -> None:
        """Save to a new directory; never silently replace a previous run."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("daily_summary", "daily_support", "opinion_distribution",
                     "candidate_results", "final_state", "sampled_state"):
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)
        manifest = {**self.metadata, "turnout": self.turnout, "diagnostics": self.diagnostics}
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fingerprint(table: pd.DataFrame) -> str:
    canonical = table.reindex(sorted(table.columns), axis=1)
    return hashlib.sha256(canonical.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _code_revision() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    # Include hashes because an uncommitted working tree is not identified by HEAD.
    files = sorted((root / "election_sim").rglob("*.py"))
    files += sorted((root / "constituency_loader").glob("*.py"))
    return {"revision": revision, "dirty": dirty,
            "source_sha256": {str(p.relative_to(root)).replace("\\", "/"):
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def _scenario(config: SimulationConfig, actions: list[CampaignAction], has_events: bool) -> str:
    types = {action.message_type for action in actions}
    if "smear" in types:
        scenario = "smear_campaign"
    elif "ordinary" in types:
        scenario = "ordinary_campaign"
    else:
        scenario = "frozen" if config.lambda_social == 0 else "social_only"
    return f"{scenario}_with_events" if has_events and config.lambda_event > 0 else scenario


def simulate(config: SimulationConfig, data: dict, seed: int = 42,
             campaign_actions: list[CampaignAction] | None = None) -> RunResult:
    """Run the social baseline with optional ordinary and smear campaigns.

    Uses canonical ID order so reordering CSV rows does not change the run.
    Scientific outputs reproduce; the manifest's creation timestamp may differ.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    voters = data["voters"].sort_values("voter_id").reset_index(drop=True).copy()
    candidates = data["candidates"].sort_values("candidate_id").reset_index(drop=True).copy()
    edges = data["social_network"].sort_values(
        ["target_voter_id", "source_voter_id"]).reset_index(drop=True).copy()
    if voters.empty or candidates.empty:
        raise ValueError("a simulation needs at least one voter and candidate")
    for name, table in (("voters", voters), ("candidates", candidates)):
        if not table.constituency_id.eq(config.constituency_id).all():
            raise ValueError(f"{name} contains a different constituency")
        key = "voter_id" if name == "voters" else "candidate_id"
        if table[key].isna().any() or table[key].duplicated().any():
            raise ValueError(f"{key} must be present and unique")
    for column, low, high in (("initial_opinion", -1, 1), ("susceptibility", 0, 1),
                              ("confidence_threshold", 0, 2), ("turnout_propensity", 0, 1)):
        values = voters[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= low) & (values <= high)).all():
            raise ValueError(f"{column} must be finite and in [{low}, {high}]")
    graph = build_incoming(voters, edges, config.rho)
    initial = voters.initial_opinion.to_numpy(dtype=float)
    opinions = initial.copy()
    susceptibility = np.clip(voters.susceptibility.to_numpy() * config.susceptibility_multiplier, 0, 1)
    confidence = (np.full(len(voters), config.global_confidence) if config.global_confidence is not None
                  else np.clip(voters.confidence_threshold.to_numpy() * config.confidence_multiplier, 0, 2))
    actions = check_actions(campaign_actions or [], config.days, set(candidates.candidate_id))
    if actions and config.lambda_campaign == 0:
        raise ValueError("campaign actions were given but lambda_campaign is 0, so they would do nothing")
    names = ("campaign", "shocks", "turnout", "voting")
    uses_communities = any(action.target_policy == "community" for action in actions)
    # Keep established campaign/social/turnout/voting streams and the
    # community stream stable; reserve a separate stream for event directions.
    children = np.random.SeedSequence(seed).spawn(6)
    child_seeds = {name: [int(v) for v in child.generate_state(4)]
                   for name, child in zip(names, children[:len(names)])}
    if uses_communities:
        child_seeds["community"] = [int(v) for v in children[4].generate_state(4)]
    child_seeds["events"] = [int(v) for v in children[5].generate_state(4)]
    rng = {name: np.random.default_rng(value) for name, value in child_seeds.items()}
    communities = (assign_communities(voters, edges, rng["community"])
                   if uses_communities else None)
    candidate_ids = candidates.candidate_id.tolist()
    events = prepare_events(data.get("events", pd.DataFrame()), voters, candidate_ids,
                            child_seeds["events"], config.days, config.constituency_id,
                            config.event_memory_half_life_days,
                            config.event_reminder_probability)
    candidate_positions = dict(zip(candidate_ids, candidates[[
        "position_employment", "position_economy", "position_education",
        "position_healthcare", "position_agriculture"]].mean(axis=1)))
    recipients = {
        action.action_id: select_recipients(
            action, voters, edges, action_rng(child_seeds["campaign"], action.action_id),
            susceptibility=susceptibility, communities=communities)
        for action in actions
    }
    influence_scores = outgoing_influence_scores(voters, edges)
    weighted_out_degree = np.divide(
        influence_scores, voters.influence_power.to_numpy(dtype=float),
        out=np.zeros(len(voters), dtype=float),
        where=voters.influence_power.to_numpy(dtype=float) > 0)
    audience_summaries = []
    for action in actions:
        selected = recipients[action.action_id]
        requested = budget(len(voters), action.reach_fraction)
        audience = voters.iloc[selected]
        group_count = (int(GROUP_MATCHERS[action.target_group](voters).sum())
                       if action.target_policy == "group" else None)
        community_id = (int(communities[selected[0]]) if action.target_policy == "community" and len(selected)
                        else action.community_id)
        community_count = (int(np.sum(communities == community_id))
                           if action.target_policy == "community" and community_id is not None else None)
        audience_summaries.append({
            "action_id": action.action_id,
            "target_policy": action.target_policy,
            "target_group": action.target_group,
            "community_id": community_id,
            "requested_recipients": requested,
            "selected_recipients": len(selected),
            "shortfall": requested - len(selected),
            "population_coverage": len(selected) / len(voters),
            "eligible_group_population": group_count,
            "eligible_community_population": community_count,
            "mean_susceptibility": (float(audience.susceptibility.mean()) if len(audience) else None),
            "mean_effective_susceptibility": (float(susceptibility[selected].mean()) if len(selected) else None),
            "mean_influence_power": (float(audience.influence_power.mean()) if len(audience) else None),
            "mean_weighted_out_degree": (float(weighted_out_degree[selected].mean()) if len(selected) else None),
            "mean_influence_score": (float(influence_scores[selected].mean()) if len(selected) else None),
        })
    # Smears lower utility; event signals can raise or lower candidate utility.
    penalty = np.zeros((len(voters), len(candidate_ids)))
    event_utility = np.zeros_like(penalty)
    summaries, support_rows, distributions, sampled_rows = [], [], [], []
    # A deterministic systematic ID sample for view state, not an electorate estimate.
    sample_indices = np.linspace(0, len(voters) - 1, min(100, len(voters)), dtype=int)
    bins = np.linspace(-1, 1, 21)
    support = candidate_support(opinions, candidates, config.b_pop, config.b_ideol,
                                config.tau, penalty, event_utility)
    applied_event_days = 0
    event_exposures = 0
    event_memory_updates = 0
    event_reminders = 0
    withdrawn_candidates = set()
    event_rows = []
    for day in range(config.days + 1):
        previous = opinions.copy()
        if day:
            opinions, _ = social_step(opinions, susceptibility, confidence, graph, config.lambda_social)
            # Actions apply in action_id order (check_actions sorts them).
            for action in actions:
                if not action.active_on(day):
                    continue
                if action.message_type == "ordinary":
                    opinions = apply_ordinary(opinions, susceptibility, recipients[action.action_id],
                                              float(candidate_positions[action.candidate_id]),
                                              action, config.lambda_campaign)
                else:
                    penalty = apply_smear(penalty, susceptibility, recipients[action.action_id],
                                          candidate_ids.index(action.target_candidate_id),
                                          action, config.lambda_campaign)
            active_events = [event for event in events if event_active_on(event, day)]
            if config.lambda_event > 0:
                for event in events:
                    if not event["occurs"]:
                        continue
                    candidate_id = event["candidate_id"]
                    if (event["structural_effect"] == "candidate_withdrawal"
                            and day == event["start_day"]):
                        if (candidate_id not in withdrawn_candidates
                                and len(withdrawn_candidates) == len(candidate_ids) - 1):
                            raise ValueError("event schedule withdraws every candidate")
                        withdrawn_candidates.add(candidate_id)
                        event_utility[:, candidate_ids.index(candidate_id)] = -1e6
                        event_rows.append({"day": day, "event_id": event["event_id"],
                                           "candidate_id": candidate_id,
                                           "effect": "candidate_withdrawal"})

                    # After the event's active window, people forget the
                    # signal exponentially. Zero half-life means it persists.
                    end_day = event["start_day"] + event["duration"] - 1
                    if day > end_day and event["memory_half_life_days"] > 0:
                        decay = 0.5 ** (1 / event["memory_half_life_days"])
                        old = event["contribution"].copy()
                        event["contribution"] *= decay
                        delta = event["contribution"] - old
                        if candidate_id is not None:
                            event_utility[:, candidate_ids.index(candidate_id)] += (
                                delta * event["candidate_effect"])
                        else:
                            opinions = np.clip(opinions + delta * event["direction"], -1, 1)
                        event_memory_updates += 1
                        event_rows.append({"day": day, "event_id": event["event_id"],
                                           "candidate_id": event["associated_candidate_id"],
                                           "effect": "memory_decay"})

                    if event_active_on(event, day):
                        strength = (config.lambda_event * event["intensity"] * event["credibility"]
                                    / event["duration"])
                        delta = strength * event["weights"]
                        event["contribution"] += delta
                        if candidate_id is not None:
                            event_utility[:, candidate_ids.index(candidate_id)] += (
                                delta * event["candidate_effect"])
                        else:
                            opinions = np.clip(opinions + delta * event["direction"], -1, 1)
                        applied_event_days += 1
                        event_exposures += len(event["audience"])
                        event_rows.append({"day": day, "event_id": event["event_id"],
                                           "source_type": event.get("source_type"),
                                           "candidate_id": event["associated_candidate_id"],
                                           "audience_size": len(event["audience"]),
                                           "mean_signal_strength": float(strength * event["weights"].mean()),
                                           "effect": "active_signal"})
                    elif (day > end_day and event["reminder_probability"] > 0
                          and event["reminder_rng"].random() < event["reminder_probability"]):
                        strength = (config.lambda_event * event["intensity"] * event["credibility"]
                                    * config.event_reminder_strength)
                        delta = strength * event["weights"]
                        event["contribution"] += delta
                        if candidate_id is not None:
                            event_utility[:, candidate_ids.index(candidate_id)] += (
                                delta * event["candidate_effect"])
                        else:
                            opinions = np.clip(opinions + delta * event["direction"], -1, 1)
                        event_reminders += 1
                        event_rows.append({"day": day, "event_id": event["event_id"],
                                           "candidate_id": event["associated_candidate_id"],
                                           "effect": "reminder"})
            support = candidate_support(opinions, candidates, config.b_pop, config.b_ideol,
                                        config.tau, penalty, event_utility)
        # Counts reflect the state logged for this day; the next update uses it.
        eligible = int((np.abs(opinions[graph.sources] - opinions[graph.targets]) <=
                        confidence[graph.targets]).sum())
        quantiles = np.quantile(opinions, [0.05, 0.25, 0.5, 0.75, 0.95])
        active_actions = [action for action in actions if action.active_on(day)]
        active_events = [event for event in events if event_active_on(event, day)]
        summaries.append({"day": day, "opinion_mean": float(opinions.mean()),
                          "opinion_sd": float(opinions.std()),
                          "polarization_variance": float(opinions.var()),
                          **dict(zip(("q05", "q25", "q50", "q75", "q95"), map(float, quantiles))),
                          "eligible_edges": eligible,
                          "active_campaign_actions": len(active_actions),
                          "campaign_exposures_today": int(sum(len(recipients[action.action_id])
                                                               for action in active_actions)),
                          "active_events": len(active_events),
                          "event_memory_updates_today": int(sum(
                              row["day"] == day and row.get("effect") == "memory_decay"
                              for row in event_rows)),
                          "event_reminders_today": int(sum(
                              row["day"] == day and row.get("effect") == "reminder"
                              for row in event_rows)),
                          "event_exposures_today": int(sum(len(event["audience"])
                                                            for event in active_events)),
                          "mean_absolute_daily_movement": float(np.abs(opinions - previous).mean()),
                          "mean_absolute_initial_movement": float(np.abs(opinions - initial).mean())})
        for candidate_id, value in zip(candidate_ids, support.mean(axis=0)):
            support_rows.append({"day": day, "candidate_id": candidate_id, "mean_support": float(value)})
        leaders = support[sample_indices].argmax(axis=1)
        for index, leader in zip(sample_indices, leaders):
            sampled_rows.append({"day": day, "voter_id": voters.voter_id.iloc[index],
                                 "opinion": float(opinions[index]),
                                 "leading_candidate_id": candidate_ids[leader]})
        counts, _ = np.histogram(opinions, bins=bins)
        for left, right, count in zip(bins[:-1], bins[1:], counts):
            distributions.append({"day": day, "bin_left": float(left), "bin_right": float(right),
                                  "count": int(count)})
    results, turnout = election_result(voters.turnout_propensity.to_numpy(), support, candidate_ids,
                                       rng["turnout"], rng["voting"], config.vote_rule)
    results["final_mean_support"] = support.mean(axis=0)
    state = pd.DataFrame({"voter_id": voters.voter_id, "initial_opinion": initial,
                          "final_opinion": opinions})
    for index, candidate_id in enumerate(candidate_ids):
        state[f"support_{candidate_id}"] = support[:, index]
    diagnostics = {**data.get("diagnostics", {}), "voter_count": len(voters),
                   "candidate_count": len(candidates), "internal_edge_count": len(edges),
                   "no_incoming_edges": int((np.diff(graph.offsets) == 0).sum()),
                   "campaign_actions_loaded": len(actions),
                   "campaign_exposures": int(sum(len(recipients[action.action_id]) * action.duration
                                                  for action in actions)),
                   "campaigns_applied": int(sum(action.active_on(day)
                                                  for action in actions for day in range(1, config.days + 1))),
                   "campaign_audiences": audience_summaries,
                   "community_count": (int(np.unique(communities).size) if communities is not None else 0),
                   "community_sizes": ([int(size) for size in np.unique(communities, return_counts=True)[1]]
                                       if communities is not None else []),
                   "events_loaded": len(events), "events_applied": applied_event_days,
                   "unique_events_applied": len({row["event_id"] for row in event_rows}),
                   "events_occurred": sum(event["occurs"] for event in events),
                   "events_suppressed": sum(not event["occurs"] for event in events),
                   "event_memory_updates": event_memory_updates,
                   "event_reminders": event_reminders,
                   "withdrawn_candidates": sorted(withdrawn_candidates),
                   "event_exposures": event_exposures, "event_applications": event_rows,
                   "loaded_events_not_applied": (len(events) if config.lambda_event == 0 else
                                                  sum(not event["occurs"] for event in events))}
    metadata = {"synthetic": True, "claim": "Synthetic scenario; not an empirically validated forecast",
                "scenario": _scenario(config, actions, bool(events)),
                "config": asdict(config), "master_seed": seed, "child_seeds": child_seeds,
                "campaign_actions": [asdict(action) for action in actions],
                "event_model": "Active-day signals apportioned over duration; effects decay by configurable half-life and can be reminded; withdrawals are structural",
                "rng": "numpy.default_rng (PCG64); canonical voter/candidate ID order",
                "view_sample": "Up to 100 systematically spaced canonical voter IDs; internal visualization only, not a representative sample",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "selected_table_sha256": {"voters": _fingerprint(voters), "candidates": _fingerprint(candidates),
                                          "social_network": _fingerprint(edges)},
                "code": _code_revision()}
    return RunResult(pd.DataFrame(summaries), pd.DataFrame(support_rows), pd.DataFrame(distributions),
                     results, state, pd.DataFrame(sampled_rows), turnout, diagnostics, metadata)
