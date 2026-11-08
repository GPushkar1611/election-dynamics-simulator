"""The daily simulation loop (plan section 3.7); see simulate()."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .agents import CandidateAgent, check_agents, run_poll
from .campaign.effects import (BELIEVES, GROUP_MATCHERS, REJECTS, UNAWARE, action_rng,
                               apply_ordinary, apply_smear, assign_communities, budget,
                               message_position, outgoing_influence_scores,
                               select_recipients, spread_smears)
from .campaign.models import CampaignAction, check_actions
from .config import SimulationConfig
from .dynamics import social_step
from .events import active_on as event_active_on
from .network import build_incoming
from .news import NewsAgent
from .support import candidate_support
from .voting import election_result, mobilized_propensity


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
    history: dict | None = None  # full per-day arrays; only when record_history=True

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
        if self.history is not None:
            save_history(self.history, directory)


# Arrays stored in history.npz. The timeline goes to history_timeline.json.
HISTORY_ARRAYS = ("voter_ids", "candidate_ids", "action_ids", "opinions", "support",
                  "smear_belief", "campaign_exposed", "event_exposed")


def save_history(history: dict, directory: str | Path) -> None:
    """Write the per-day arrays to history.npz and the timeline to JSON."""
    directory = Path(directory)
    np.savez_compressed(directory / "history.npz",
                        **{name: history[name] for name in HISTORY_ARRAYS})
    (directory / "history_timeline.json").write_text(
        json.dumps(history["timeline"], indent=2) + "\n", encoding="utf-8")


def load_history(directory: str | Path) -> dict:
    """Read a history saved by RunResult.save back into the same dict shape."""
    directory = Path(directory)
    with np.load(directory / "history.npz") as arrays:
        history = {name: arrays[name] for name in arrays.files}
    history["timeline"] = json.loads(
        (directory / "history_timeline.json").read_text(encoding="utf-8"))
    return history


def _timeline(actions: list[CampaignAction], events: list[dict],
              event_rows: list[dict]) -> list[dict]:
    """Campaigns and events in day order, for labelling plots and videos."""
    timeline = []
    for action in actions:
        target = f" -> {action.target_candidate_id}" if action.target_candidate_id else ""
        timeline.append({"day": action.start_day,
                         "end_day": action.start_day + action.duration - 1,
                         "kind": f"campaign_{action.message_type}", "id": action.action_id,
                         "label": f"{action.message_type} by {action.candidate_id}{target}",
                         "candidate_id": action.candidate_id})
    for event in events:
        if event["occurs"]:
            timeline.append({"day": event["start_day"],
                             "end_day": event["start_day"] + event["duration"] - 1,
                             "kind": event.get("news_kind") or "event", "id": event["event_id"],
                             "label": event["event_type"] or event["event_id"],
                             "candidate_id": event["associated_candidate_id"]})
    for row in event_rows:
        if row["effect"] in ("candidate_withdrawal", "reminder"):
            timeline.append({"day": row["day"], "end_day": row["day"],
                             "kind": f"event_{row['effect']}", "id": row["event_id"],
                             "label": row["effect"].replace("_", " "),
                             "candidate_id": row["candidate_id"]})
    return sorted(timeline, key=lambda item: (item["day"], item["kind"], item["id"]))


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


def _scenario(config: SimulationConfig, actions: list[CampaignAction], has_agents: bool,
              has_events: bool) -> str:
    types = {action.message_type for action in actions}
    if has_agents:
        scenario = "agent_campaign"
    elif "smear" in types:
        scenario = "smear_campaign"
    elif "misleading" in types:
        scenario = "misleading_campaign"
    elif "ordinary" in types:
        scenario = "ordinary_campaign"
    else:
        scenario = "frozen" if config.lambda_social == 0 else "social_only"
    return f"{scenario}_with_events" if has_events and config.lambda_event > 0 else scenario


# Child random streams, one per purpose. Index 1 held a stream that is no
# longer used; keeping fixed indices keeps every other stream (and so results
# from earlier versions) unchanged.
STREAM_INDEX = {"campaign": 0, "turnout": 2, "voting": 3, "community": 4,
                "events": 5, "polls": 6, "spread": 7}


def _child_seeds(seed: int) -> dict:
    children = np.random.SeedSequence(seed).spawn(max(STREAM_INDEX.values()) + 1)
    return {name: [int(value) for value in children[index].generate_state(4)]
            for name, index in STREAM_INDEX.items()}


def _audience_summary(action: CampaignAction, selected: np.ndarray, voters: pd.DataFrame,
                      susceptibility: np.ndarray, communities: np.ndarray | None,
                      influence_scores: np.ndarray, weighted_out_degree: np.ndarray) -> dict:
    """Aggregate description of who an action reached (no voter IDs)."""
    requested = budget(len(voters), action.reach_fraction)
    audience = voters.iloc[selected]
    group_count = (int(GROUP_MATCHERS[action.target_group](voters).sum())
                   if action.target_policy == "group" else None)
    community_id = (int(communities[selected[0]]) if action.target_policy == "community" and len(selected)
                    else action.community_id)
    community_count = (int(np.sum(communities == community_id))
                       if action.target_policy == "community" and community_id is not None else None)
    return {
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
    }


def simulate(config: SimulationConfig, data: dict, seed: int = 42,
             campaign_actions: list[CampaignAction] | None = None,
             record_history: bool = False,
             agents: list[CandidateAgent] | None = None,
             news_agent: NewsAgent | None = None) -> RunResult:
    """Run one synthetic election.

    campaign_actions: scripted actions fixed in advance (campaign_actions/*.json).
    agents: CandidateAgents that read a poll on their decision days and choose
        actions during the run (campaign_agents/*.json).
    news_agent: optional CSV/DataFrame scheduler; otherwise uses data events and
        config.random_news_rate. Public polls are saved in diagnostics.
    record_history: also keep every voter's daily state (RunResult.history).
        Recording only copies values; it never changes the simulation.

    Each day: (1) agents poll yesterday's support and choose actions,
    (2) voters influence each other, (3) campaign messages apply,
    (4) smear rumours spread, (5) scheduled events apply, (6) support is
    recomputed. After the last day, turnout and votes are sampled.

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
    candidate_ids = candidates.candidate_id.tolist()
    actions = check_actions(campaign_actions or [], config.days, set(candidate_ids))
    agents = check_agents(agents or [], set(candidate_ids))
    if (actions or agents) and config.lambda_campaign == 0:
        raise ValueError("campaign actions or agents were given but lambda_campaign is 0, "
                         "so they would do nothing")

    child_seeds = _child_seeds(seed)
    turnout_rng, vote_rng, poll_rng, spread_rng = (
        np.random.default_rng(child_seeds[name]) for name in ("turnout", "voting", "polls", "spread"))
    news_agent = news_agent or NewsAgent(random_events_per_day=config.random_news_rate)
    events = news_agent.prepare(data.get("events", pd.DataFrame()), voters, candidate_ids,
                                child_seeds["events"], config)
    campaign_exposure = np.zeros(len(voters), dtype=int)
    news_exposure = np.zeros(len(voters), dtype=int)
    public_polls = []
    platforms = dict(zip(candidate_ids, candidates[[
        "position_employment", "position_economy", "position_education",
        "position_healthcare", "position_agriculture"]].mean(axis=1)))
    # Social ties as index arrays, for rumour spreading.
    voter_index = pd.Index(voters.voter_id)
    edge_sources = voter_index.get_indexer(edges.source_voter_id)
    edge_targets = voter_index.get_indexer(edges.target_voter_id)
    edge_weights = edges.edge_weight.to_numpy(dtype=float)

    # Communities are only inferred if some action targets one.
    communities = None

    def choose_recipients(action: CampaignAction) -> np.ndarray:
        nonlocal communities
        if action.target_policy == "community" and communities is None:
            communities = assign_communities(voters, edges,
                                             np.random.default_rng(child_seeds["community"]))
        return select_recipients(action, voters, edges,
                                 action_rng(child_seeds["campaign"], action.action_id),
                                 susceptibility=susceptibility, communities=communities)

    recipients = {action.action_id: choose_recipients(action) for action in actions}
    budget_left = {agent.candidate_id: agent.budget for agent in agents}
    last_poll = {}  # the poll each agent saw at its previous decision
    polls, decisions = [], []

    # Smear claims: belief state and strength per voter x candidate. Only
    # believers' strength counts as a utility penalty.
    smear_belief = np.full((len(voters), len(candidate_ids)), UNAWARE, dtype=np.int8)
    smear_strength = np.zeros((len(voters), len(candidate_ids)))
    penalty = np.zeros_like(smear_strength)
    event_utility = np.zeros_like(penalty)  # event signals can raise or lower utility
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
    if record_history:
        steps, voter_count = config.days + 1, len(voters)
        opinion_history = np.zeros((steps, voter_count), dtype=np.float32)
        support_history = np.zeros((steps, voter_count, len(candidate_ids)), dtype=np.float32)
        belief_history = np.zeros((steps, voter_count, len(candidate_ids)), dtype=np.int8)
        event_history = np.zeros((steps, voter_count), dtype=bool)
    for day in range(config.days + 1):
        previous = opinions.copy()
        news_today = np.zeros(len(voters), dtype=bool)
        campaign_today = np.zeros(len(voters), dtype=bool)
        if day:
            # (1) Agents read a poll of yesterday's support and choose actions.
            deciding = [agent for agent in agents if agent.decides_on(day)]
            if deciding:
                poll = run_poll(support, voters, candidate_ids, poll_rng, day,
                                config.poll_sample_size, config.poll_noise)
                polls.append(poll)
                for agent in deciding:
                    action = agent.decide(day, config.days, poll, last_poll.get(agent.candidate_id))
                    last_poll[agent.candidate_id] = poll
                    if action.action_id in recipients:
                        raise ValueError(f"duplicate campaign action_id {action.action_id}")
                    chosen = choose_recipients(action)
                    cost = len(chosen) * action.duration
                    taken = cost <= budget_left[agent.candidate_id]
                    if taken:
                        budget_left[agent.candidate_id] -= cost
                        actions.append(action)
                        recipients[action.action_id] = chosen
                    decisions.append({
                        "day": day, "candidate_id": agent.candidate_id, "strategy": agent.strategy,
                        "action_id": action.action_id, "message_type": action.message_type,
                        "target_policy": action.target_policy, "target_group": action.target_group,
                        "target_candidate_id": action.target_candidate_id,
                        "poll_share": poll["shares"][agent.candidate_id],
                        "recipients": len(chosen), "cost": cost, "taken": taken,
                        "budget_left": budget_left[agent.candidate_id]})
                actions.sort(key=lambda item: item.action_id)

            # (2) Voters influence each other.
            opinions, _ = social_step(opinions, susceptibility, confidence, graph, config.lambda_social)

            # (3) Campaign messages, in action_id order.
            for action in actions:
                if not action.active_on(day):
                    continue
                chosen = recipients[action.action_id]
                campaign_today[chosen] = True
                if action.message_type == "smear":
                    target = candidate_ids.index(action.target_candidate_id)
                    smear_strength = apply_smear(smear_strength, susceptibility, chosen, target,
                                                 action, config.lambda_campaign)
                    smear_belief[chosen, target] = BELIEVES
                else:  # ordinary or misleading
                    position = message_position(action, float(platforms[action.candidate_id]),
                                                opinions, chosen)
                    opinions = apply_ordinary(opinions, susceptibility, chosen, position,
                                              action, config.lambda_campaign)

            # (4) Smear rumours spread along social ties.
            if config.smear_spread_probability > 0:
                smear_belief, smear_strength = spread_smears(
                    smear_belief, smear_strength, edge_sources, edge_targets, edge_weights,
                    susceptibility, config.smear_spread_probability, spread_rng)
            penalty = np.where(smear_belief == BELIEVES, smear_strength, 0.0)

            # (5) Scheduled events.
            if config.lambda_event > 0:
                for event in events:
                    if not event["occurs"]:
                        continue
                    if event.get("news_kind"):
                        if event_active_on(event, day):
                            audience = event["audience"]
                            news_today[audience] = True
                            applied_event_days += 1
                            event_exposures += len(audience)
                            accepted = 0
                            if event["news_kind"] == "fact_check":
                                target = candidate_ids.index(event["candidate_id"])
                                believers = audience[smear_belief[audience, target] == BELIEVES]
                                corrected = believers[event["news_rng"].random(len(believers)) < event["credibility"]]
                                smear_belief[corrected, target] = REJECTS
                                accepted = len(corrected)
                            event_rows.append({"day": day, "event_id": event["event_id"],
                                "candidate_id": event["associated_candidate_id"],
                                "effect": event["news_kind"], "audience_size": len(audience),
                                "accepted": accepted})
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
                        news_today[event["audience"]] = True
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
                        news_today[event["audience"]] = True
                        event_reminders += 1
                        event_rows.append({"day": day, "event_id": event["event_id"],
                                           "candidate_id": event["associated_candidate_id"],
                                           "effect": "reminder"})

            campaign_exposure += campaign_today
            news_exposure += news_today
            penalty = np.where(smear_belief == BELIEVES, smear_strength, 0.0)
            # (6) Recompute candidate support.
            support = candidate_support(opinions, candidates, config.b_pop, config.b_ideol,
                                        config.tau, penalty, event_utility)
            for event in events:
                if (config.lambda_event > 0 and event.get("news_kind") == "poll_release"
                        and event_active_on(event, day)):
                    published = run_poll(support, voters, candidate_ids, event["news_rng"], day,
                                         config.poll_sample_size, config.poll_noise)
                    public_polls.append({**published, "event_id": event["event_id"],
                                         "announcement": "public"})
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
                          "smear_believers": int((smear_belief == BELIEVES).any(axis=1).sum()),
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
        if record_history:
            opinion_history[day] = opinions
            support_history[day] = support
            belief_history[day] = smear_belief
            event_history[day] = news_today
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
    turnout_probability = mobilized_propensity(voters.turnout_propensity.to_numpy(),
        campaign_exposure, news_exposure, config.gamma_mob, config.gamma_event)
    results, turnout = election_result(turnout_probability, support, candidate_ids,
                                       turnout_rng, vote_rng, config.vote_rule)
    results["final_mean_support"] = support.mean(axis=0)
    state = pd.DataFrame({"voter_id": voters.voter_id, "initial_opinion": initial,
                          "final_opinion": opinions, "campaign_exposure": campaign_exposure,
                          "event_exposure": news_exposure, "turnout_probability": turnout_probability})
    for index, candidate_id in enumerate(candidate_ids):
        state[f"support_{candidate_id}"] = support[:, index]
        state[f"smear_belief_{candidate_id}"] = smear_belief[:, index]

    influence_scores = outgoing_influence_scores(voters, edges)
    powers = voters.influence_power.to_numpy(dtype=float)
    weighted_out_degree = np.divide(influence_scores, powers, out=np.zeros(len(voters)),
                                    where=powers > 0)
    audience_summaries = [_audience_summary(action, recipients[action.action_id], voters,
                                            susceptibility, communities, influence_scores,
                                            weighted_out_degree) for action in actions]
    diagnostics = {**data.get("diagnostics", {}), "voter_count": len(voters),
                   "candidate_count": len(candidates), "internal_edge_count": len(edges),
                   "no_incoming_edges": int((np.diff(graph.offsets) == 0).sum()),
                   "campaign_actions_loaded": len(actions),
                   "campaign_exposures": int(sum(len(recipients[action.action_id]) * action.duration
                                                  for action in actions)),
                   "campaigns_applied": int(sum(action.active_on(day)
                                                  for action in actions for day in range(1, config.days + 1))),
                   "campaign_audiences": audience_summaries,
                   "agent_decisions": decisions,
                   "agent_budget_left": budget_left,
                   "polls": polls, "public_polls": public_polls,
                   "smear_believers": {candidate_id: int((smear_belief[:, index] == BELIEVES).sum())
                                       for index, candidate_id in enumerate(candidate_ids)},
                   "smear_rejecters": {candidate_id: int((smear_belief[:, index] == REJECTS).sum())
                                       for index, candidate_id in enumerate(candidate_ids)},
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
                "scenario": _scenario(config, actions, bool(agents), bool(events)),
                "config": asdict(config), "master_seed": seed, "child_seeds": child_seeds,
                "campaign_actions": [asdict(action) for action in actions],
                "agents": [asdict(agent) for agent in agents],
                "news_agent": {"random_events_per_day": news_agent.random_events_per_day},
                "event_model": "Active-day signals apportioned over duration; effects decay by configurable half-life and can be reminded; withdrawals are structural",
                "rng": "numpy.default_rng (PCG64); canonical voter/candidate ID order",
                "view_sample": "Up to 100 systematically spaced canonical voter IDs; internal visualization only, not a representative sample",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "selected_table_sha256": {"voters": _fingerprint(voters), "candidates": _fingerprint(candidates),
                                          "social_network": _fingerprint(edges)},
                "code": _code_revision(),
                "history_recorded": record_history}
    history = None
    if record_history:
        # Campaign exposure follows from each action's recipients and active days.
        campaign_history = np.zeros((config.days + 1, len(voters), len(actions)), dtype=bool)
        for column, action in enumerate(actions):
            for day in range(action.start_day, action.start_day + action.duration):
                campaign_history[day, recipients[action.action_id], column] = True
        history = {
            "voter_ids": voters.voter_id.to_numpy(dtype=str),
            "candidate_ids": np.array(candidate_ids, dtype=str),
            "action_ids": np.array([action.action_id for action in actions], dtype=str),
            "opinions": opinion_history,             # [day, voter]
            "support": support_history,              # [day, voter, candidate]
            "smear_belief": belief_history,          # [day, voter, candidate] 0/1/2
            "campaign_exposed": campaign_history,    # [day, voter, action]
            "event_exposed": event_history,          # [day, voter]
            "timeline": _timeline(actions, events if config.lambda_event > 0 else [], event_rows),
        }
    return RunResult(pd.DataFrame(summaries), pd.DataFrame(support_rows), pd.DataFrame(distributions),
                     results, state, pd.DataFrame(sampled_rows), turnout, diagnostics, metadata,
                     history)
