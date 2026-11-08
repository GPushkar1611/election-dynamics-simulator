"""Small paired counterfactuals for supplied news/event schedules."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import CandidateAgent
from .campaign.effects import GROUP_MATCHERS
from .campaign.models import CampaignAction
from .config import SimulationConfig
from .simulation import RunResult, simulate


@dataclass
class PairedEventResult:
    no_shock: RunResult
    with_shock: RunResult
    candidate_effects: pd.DataFrame
    opinion_effect: dict
    group_effects: pd.DataFrame


@dataclass
class EventSeedSweepResult:
    per_seed: pd.DataFrame
    per_seed_candidate_effects: pd.DataFrame
    candidate_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    per_seed_group_effects: pd.DataFrame
    group_summary: pd.DataFrame
    metadata: dict

    def save(self, output_dir: str | Path) -> None:
        """Save compact paired outputs into a new directory."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        self.per_seed.to_csv(directory / "per_seed.csv", index=False)
        self.per_seed_candidate_effects.to_csv(
            directory / "per_seed_candidate_effects.csv", index=False)
        self.candidate_summary.to_csv(directory / "candidate_summary.csv", index=False)
        self.overall_summary.to_csv(directory / "overall_summary.csv", index=False)
        self.per_seed_group_effects.to_csv(directory / "per_seed_group_effects.csv", index=False)
        self.group_summary.to_csv(directory / "group_summary.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class EventFactorialResult:
    per_seed_arm: pd.DataFrame
    per_seed_candidate_effects: pd.DataFrame
    arm_summary: pd.DataFrame
    candidate_summary: pd.DataFrame
    per_seed_group_effects: pd.DataFrame
    group_summary: pd.DataFrame
    metadata: dict

    def save(self, output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        self.per_seed_arm.to_csv(directory / "per_seed_arm.csv", index=False)
        self.per_seed_candidate_effects.to_csv(
            directory / "per_seed_candidate_effects.csv", index=False)
        self.arm_summary.to_csv(directory / "arm_summary.csv", index=False)
        self.candidate_summary.to_csv(directory / "candidate_summary.csv", index=False)
        self.per_seed_group_effects.to_csv(directory / "per_seed_group_effects.csv", index=False)
        self.group_summary.to_csv(directory / "group_summary.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class SupportSensitivityResult:
    per_seed: pd.DataFrame
    per_seed_candidate: pd.DataFrame
    per_seed_agent_decisions: pd.DataFrame
    summary: pd.DataFrame
    candidate_summary: pd.DataFrame
    metadata: dict

    def save(self, output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        self.per_seed.to_csv(directory / "per_seed.csv", index=False)
        self.per_seed_candidate.to_csv(directory / "per_seed_candidate.csv", index=False)
        self.per_seed_agent_decisions.to_csv(
            directory / "per_seed_agent_decisions.csv", index=False)
        self.summary.to_csv(directory / "summary.csv", index=False)
        self.candidate_summary.to_csv(directory / "candidate_summary.csv", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _group_effects(control_state: pd.DataFrame, event_state: pd.DataFrame,
                   voters: pd.DataFrame, **keys) -> pd.DataFrame:
    """Summarize paired opinion and candidate-support changes by known voter group."""
    group_masks = {"General": pd.Series(True, index=voters.index).to_numpy()}
    group_masks.update({name: matcher(voters) for name, matcher in GROUP_MATCHERS.items()})
    control = control_state.set_index("voter_id")
    changed = event_state.set_index("voter_id").reindex(control.index)
    voter_ids = voters.voter_id.to_numpy()
    opinion_delta = changed.final_opinion.to_numpy() - control.final_opinion.to_numpy()
    candidate_ids = [column.removeprefix("support_") for column in control
                     if column.startswith("support_")]
    rows = []
    for group_name, mask in group_masks.items():
        selected = control.index.isin(voter_ids[mask])
        count = int(selected.sum())
        if count == 0:
            continue
        for candidate_id in candidate_ids:
            support_delta = (changed.loc[selected, f"support_{candidate_id}"].to_numpy()
                             - control.loc[selected, f"support_{candidate_id}"].to_numpy())
            rows.append({**keys, "group": group_name, "voter_count": count,
                         "candidate_id": candidate_id,
                         "mean_opinion_difference": float(opinion_delta[selected].mean()),
                         "mean_absolute_opinion_difference": float(abs(opinion_delta[selected]).mean()),
                         "mean_support_difference": float(support_delta.mean())})
    return pd.DataFrame(rows)


def _summarize_group_effects(per_group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if per_group.empty:
        return pd.DataFrame(columns=[*keys, "seed_count"])
    rows = []
    for group_key, values in per_group.groupby(keys, sort=False, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, group_key))
        row["seed_count"] = int(values.seed.nunique())
        row["voter_count"] = int(values.voter_count.iloc[0])
        for metric in ("mean_opinion_difference", "mean_absolute_opinion_difference",
                       "mean_support_difference"):
            row[f"{metric}_mean"] = float(values[metric].mean())
            row[f"{metric}_p05"] = float(values[metric].quantile(0.05))
            row[f"{metric}_p95"] = float(values[metric].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_event_shock(config, data: dict, seed: int = 42,
                        campaign_actions=None, agents=None) -> PairedEventResult:
    """Run matched no-event and event scenarios with common random streams.

    The only changed parameter is ``lambda_event``. Social, campaign, turnout,
    voting, event occurrence, and event reminder streams use the same seeds.
    The supplied schedule is retained in both runs; the control has zero event
    strength.
    """
    if config.lambda_event <= 0:
        raise ValueError("the shock run requires lambda_event > 0")
    no_shock = simulate(replace(config, lambda_event=0.0), data, seed, campaign_actions, agents=agents)
    with_shock = simulate(config, data, seed, campaign_actions, agents=agents)
    day = config.days
    control_support = no_shock.daily_support.query("day == @day").set_index("candidate_id").mean_support
    shock_support = with_shock.daily_support.query("day == @day").set_index("candidate_id").mean_support
    control_votes = no_shock.candidate_results.set_index("candidate_id").vote_share
    shock_votes = with_shock.candidate_results.set_index("candidate_id").vote_share
    candidate_effects = pd.DataFrame({
        "candidate_id": control_support.index,
        "support_no_shock": control_support.to_numpy(),
        "support_with_shock": shock_support.reindex(control_support.index).to_numpy(),
        "support_difference": (shock_support - control_support).reindex(control_support.index).to_numpy(),
        "vote_share_no_shock": control_votes.reindex(control_support.index).to_numpy(),
        "vote_share_with_shock": shock_votes.reindex(control_support.index).to_numpy(),
        "vote_share_difference": (shock_votes - control_votes).reindex(control_support.index).to_numpy(),
    })
    opinion_delta = (with_shock.final_state.final_opinion.to_numpy()
                     - no_shock.final_state.final_opinion.to_numpy())
    opinion_effect = {"mean_difference": float(opinion_delta.mean()),
                      "mean_absolute_difference": float(abs(opinion_delta).mean()),
                      "voters_changed": int((opinion_delta != 0).sum()),
                      "voter_count": len(opinion_delta)}
    group_effects = _group_effects(no_shock.final_state, with_shock.final_state,
                                   data["voters"])
    return PairedEventResult(no_shock, with_shock, candidate_effects,
                             opinion_effect, group_effects)


def run_event_seed_sweep(config, data: dict, seeds, campaign_actions=None) -> EventSeedSweepResult:
    """Run paired shock/no-shock comparisons over an explicit seed list.

    Every seed runs both conditions with common random streams. Summary bands
    use the empirical 5th and 95th percentiles across the requested seeds;
    they describe this synthetic scenario sweep, not real-world uncertainty.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
           for seed in seed_list):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    if config.lambda_event <= 0:
        raise ValueError("the event seed sweep requires lambda_event > 0")
    if data.get("events") is None or len(data["events"]) == 0:
        raise ValueError("the event seed sweep needs a nonempty event schedule")

    seed_rows = []
    candidate_rows = []
    group_rows = []
    for seed in seed_list:
        paired = compare_event_shock(config, data, seed, campaign_actions)
        no_shock, with_shock = paired.no_shock, paired.with_shock
        seed_rows.append({
            "seed": seed,
            "population": len(no_shock.final_state),
            "turnout_no_shock": no_shock.turnout["turnout"],
            "turnout_with_shock": with_shock.turnout["turnout"],
            "turnout_rate_difference": (with_shock.turnout["turnout_rate"]
                                         - no_shock.turnout["turnout_rate"]),
            "winner_no_shock": ";".join(no_shock.turnout["winner_ids"]),
            "winner_with_shock": ";".join(with_shock.turnout["winner_ids"]),
            "winner_changed": (set(no_shock.turnout["winner_ids"])
                               != set(with_shock.turnout["winner_ids"])),
            "mean_opinion_difference": paired.opinion_effect["mean_difference"],
            "mean_absolute_opinion_difference": paired.opinion_effect["mean_absolute_difference"],
            "voters_changed": paired.opinion_effect["voters_changed"],
            "events_loaded": with_shock.diagnostics["events_loaded"],
            "events_occurred": with_shock.diagnostics["events_occurred"],
            "active_day_applications": with_shock.diagnostics["events_applied"],
            "memory_updates": with_shock.diagnostics["event_memory_updates"],
            "reminders": with_shock.diagnostics["event_reminders"],
            "withdrawn_candidates": ";".join(with_shock.diagnostics["withdrawn_candidates"]),
        })
        rows = paired.candidate_effects.copy()
        rows.insert(0, "seed", seed)
        candidate_rows.extend(rows.to_dict("records"))
        group_rows.extend(paired.group_effects.assign(seed=seed).to_dict("records"))

    per_seed = pd.DataFrame(seed_rows)
    per_candidate = pd.DataFrame(candidate_rows)
    candidate_summary_rows = []
    for candidate_id, group in per_candidate.groupby("candidate_id", sort=True):
        row = {"candidate_id": candidate_id, "seed_count": len(group)}
        for metric in ("support_difference", "vote_share_difference"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(pd.Series(values).median())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_p05"] = float(pd.Series(values).quantile(0.05))
            row[f"{metric}_p95"] = float(pd.Series(values).quantile(0.95))
        candidate_summary_rows.append(row)
    candidate_summary = pd.DataFrame(candidate_summary_rows)
    per_group = pd.DataFrame(group_rows)
    group_summary = _summarize_group_effects(per_group, ["group", "candidate_id"])
    overall_summary = pd.DataFrame([{
        "seed_count": len(seed_list),
        "winner_changed_rate": float(per_seed.winner_changed.mean()),
        "mean_absolute_opinion_difference": float(per_seed.mean_absolute_opinion_difference.mean()),
        "mean_absolute_opinion_difference_p05": float(per_seed.mean_absolute_opinion_difference.quantile(0.05)),
        "mean_absolute_opinion_difference_p95": float(per_seed.mean_absolute_opinion_difference.quantile(0.95)),
        "mean_turnout_rate_difference": float(per_seed.turnout_rate_difference.mean()),
        "mean_events_occurred": float(per_seed.events_occurred.mean()),
        "mean_active_day_applications": float(per_seed.active_day_applications.mean()),
        "mean_reminders": float(per_seed.reminders.mean()),
    }])
    metadata = {
        "synthetic": True,
        "claim": "Paired synthetic scenario sweep; not empirically calibrated or a forecast",
        "config": asdict(config),
        "seeds": seed_list,
        "schedule_event_count": len(data["events"]),
        "comparison": "lambda_event on versus zero; common random streams within each seed",
        "campaign_actions": [asdict(action) for action in (campaign_actions or [])],
        "uncertainty_note": "Percentiles summarize only the specified synthetic seed sweep.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return EventSeedSweepResult(per_seed, per_candidate, candidate_summary,
                                overall_summary, per_group, group_summary, metadata)


EVENT_FACTORIAL_ARMS = (
    "no_events", "structural_only", "candidate_associated_only",
    "neutral_shocks_only", "all_nonstructural", "full_schedule",
)


def _event_arm_data(data: dict, arm: str, candidate_ids: set[str]) -> dict:
    events = data["events"].copy()
    structural = (events.get("structural_effect", pd.Series("none", index=events.index))
                  .fillna("none").eq("candidate_withdrawal"))
    named = events.candidate_id.fillna("").astype(str)
    associated = named.isin(candidate_ids) & named.ne("")
    if arm == "no_events":
        selected = events.iloc[:0].copy()
    elif arm == "structural_only":
        selected = events.loc[structural].copy()
    elif arm == "candidate_associated_only":
        selected = events.loc[associated & ~structural].copy()
    elif arm == "neutral_shocks_only":
        # Candidate names outside this constituency's ballot are neutral here.
        selected = events.loc[~associated & ~structural].copy()
    elif arm == "all_nonstructural":
        selected = events.loc[~structural].copy()
    elif arm == "full_schedule":
        selected = events.copy()
    else:
        raise ValueError(f"unknown event factorial arm {arm!r}")

    selected_ids = set(selected.event_id.astype(str))
    # If an arm omits an event that a selected row depends on, suppress that
    # child in this arm instead of silently turning it into an unconditional event.
    if "depends_on_event_id" in selected:
        missing_parent = (selected.depends_on_event_id.notna()
                          & selected.depends_on_event_id.astype(str).ne("")
                          & ~selected.depends_on_event_id.astype(str).isin(selected_ids))
        if missing_parent.any():
            if "occurrence_probability" not in selected:
                selected["occurrence_probability"] = 1.0
            selected.loc[missing_parent, "occurrence_probability"] = 0.0
            selected.loc[missing_parent, "depends_on_event_id"] = ""
    arm_data = dict(data)
    arm_data["events"] = selected.reset_index(drop=True)
    arm_data["diagnostics"] = {**data.get("diagnostics", {}),
                                "event_count": len(selected), "event_arm": arm}
    return arm_data


def run_event_factorial_sweep(config, data: dict, seeds,
                              campaign_actions=None) -> EventFactorialResult:
    """Run matched seeds across structural, candidate, neutral, and full-event arms.

    All arms share per-seed social, campaign, turnout, voting, and event RNG
    streams. Effects in every arm are compared with that seed's no-events arm.
    """
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
           for seed in seed_list):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    if config.lambda_event <= 0:
        raise ValueError("the event factorial sweep requires lambda_event > 0")
    if data.get("events") is None or len(data["events"]) == 0:
        raise ValueError("the event factorial sweep needs a nonempty event schedule")

    candidate_ids = set(data["candidates"].candidate_id.astype(str))
    seed_arm_rows, candidate_rows, group_rows = [], [], []
    for seed in seed_list:
        runs = {}
        for arm in EVENT_FACTORIAL_ARMS:
            arm_data = _event_arm_data(data, arm, candidate_ids)
            runs[arm] = simulate(config, arm_data, seed, campaign_actions)
        control = runs["no_events"]
        control_vote_share = control.candidate_results.set_index("candidate_id").vote_share
        control_mean_support = {
            candidate_id: float(control.final_state[f"support_{candidate_id}"].mean())
            for candidate_id in candidate_ids
        }
        control_opinion = control.final_state.set_index("voter_id").final_opinion

        for arm in EVENT_FACTORIAL_ARMS:
            result = runs[arm]
            opinion = result.final_state.set_index("voter_id").final_opinion
            opinion_delta = opinion.reindex(control_opinion.index).to_numpy() - control_opinion.to_numpy()
            winner_changed = (set(result.turnout["winner_ids"])
                              != set(control.turnout["winner_ids"]))
            seed_arm_rows.append({
                "seed": seed, "arm": arm,
                "winner": ";".join(result.turnout["winner_ids"]),
                "winner_changed_vs_no_events": winner_changed,
                "turnout": result.turnout["turnout"],
                "turnout_rate_difference_vs_no_events": (
                    result.turnout["turnout_rate"] - control.turnout["turnout_rate"]),
                "mean_opinion_difference_vs_no_events": float(opinion_delta.mean()),
                "mean_absolute_opinion_difference_vs_no_events": float(abs(opinion_delta).mean()),
                "voters_changed_vs_no_events": int((opinion_delta != 0).sum()),
                "events_loaded": result.diagnostics["events_loaded"],
                "events_occurred": result.diagnostics["events_occurred"],
                "events_suppressed": result.diagnostics["events_suppressed"],
                "active_day_applications": result.diagnostics["events_applied"],
                "reminders": result.diagnostics["event_reminders"],
                "withdrawn_candidates": ";".join(result.diagnostics["withdrawn_candidates"]),
            })
            vote_share = result.candidate_results.set_index("candidate_id").vote_share
            for candidate_id in sorted(candidate_ids):
                support = float(result.final_state[f"support_{candidate_id}"].mean())
                share = float(vote_share.get(candidate_id, 0.0))
                candidate_rows.append({
                    "seed": seed, "arm": arm, "candidate_id": candidate_id,
                    "mean_support": support,
                    "mean_support_difference_vs_no_events": support - control_mean_support[candidate_id],
                    "vote_share": share,
                    "vote_share_difference_vs_no_events": (
                        share - float(control_vote_share.get(candidate_id, 0.0))),
                })
            group_rows.extend(_group_effects(control.final_state, result.final_state,
                                             data["voters"], seed=seed, arm=arm)
                              .to_dict("records"))

    per_seed_arm = pd.DataFrame(seed_arm_rows)
    per_candidate = pd.DataFrame(candidate_rows)
    arm_summary_rows = []
    for arm, group in per_seed_arm.groupby("arm", sort=False):
        arm_summary_rows.append({
            "arm": arm, "seed_count": len(group),
            "winner_changed_rate_vs_no_events": float(group.winner_changed_vs_no_events.mean()),
            "mean_absolute_opinion_difference": float(group.mean_absolute_opinion_difference_vs_no_events.mean()),
            "mean_absolute_opinion_difference_p05": float(group.mean_absolute_opinion_difference_vs_no_events.quantile(0.05)),
            "mean_absolute_opinion_difference_p95": float(group.mean_absolute_opinion_difference_vs_no_events.quantile(0.95)),
            "mean_turnout_rate_difference": float(group.turnout_rate_difference_vs_no_events.mean()),
            "mean_events_occurred": float(group.events_occurred.mean()),
            "mean_active_day_applications": float(group.active_day_applications.mean()),
            "mean_reminders": float(group.reminders.mean()),
        })
    candidate_summary_rows = []
    for (arm, candidate_id), group in per_candidate.groupby(["arm", "candidate_id"], sort=False):
        summary = {"arm": arm, "candidate_id": candidate_id, "seed_count": len(group)}
        for metric in ("mean_support_difference_vs_no_events", "vote_share_difference_vs_no_events"):
            values = group[metric]
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(values.median())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_p05"] = float(values.quantile(0.05))
            summary[f"{metric}_p95"] = float(values.quantile(0.95))
        candidate_summary_rows.append(summary)
    per_group = pd.DataFrame(group_rows)
    group_summary = _summarize_group_effects(
        per_group, ["arm", "group", "candidate_id"])
    metadata = {
        "synthetic": True,
        "claim": "Matched multi-arm synthetic event sweep; not empirical validation or a forecast",
        "config": asdict(config), "seeds": seed_list,
        "campaign_actions": [asdict(action) for action in (campaign_actions or [])],
        "schedule_event_count": len(data["events"]),
        "arms": list(EVENT_FACTORIAL_ARMS),
        "control": "Each arm compared with the same-seed no_events arm",
        "dependency_note": "If an arm excludes a selected event's parent, the child is suppressed in that arm.",
        "uncertainty_note": "Percentiles summarize only the specified synthetic seed sweep.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return EventFactorialResult(per_seed_arm, per_candidate,
                                pd.DataFrame(arm_summary_rows),
                                pd.DataFrame(candidate_summary_rows),
                                per_group, group_summary, metadata)


def run_support_sensitivity_sweep(config, data: dict, seeds,
                                  dispersions=(0.0, 0.2, 0.35, 0.6, 1.0),
                                  campaign_actions=None,
                                  agents: list[CandidateAgent] | None = None) -> SupportSensitivityResult:
    """Compare scalar support with issue-aware support across profile dispersions.

    Seeds are matched across every arm. Synthetic issue offsets use the same
    base draws at every dispersion, so changing dispersion scales profiles
    without changing which issue-offset pattern a voter receives.
    """
    seed_list = list(seeds)
    dispersion_list = list(dispersions)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
           for seed in seed_list):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    if not dispersion_list:
        raise ValueError("at least one synthetic issue dispersion is required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not pd.notna(value) or not 0 <= float(value) <= 1
           for value in dispersion_list):
        raise ValueError("synthetic issue dispersions must be finite values in [0, 1]")
    dispersion_list = [float(value) for value in dispersion_list]
    if len(set(dispersion_list)) != len(dispersion_list):
        raise ValueError("synthetic issue dispersions must be unique")
    columns = [f"issue_{issue}" for issue in ("employment", "economy", "education",
                                                "healthcare", "agriculture")]
    missing = set(columns).difference(data["voters"].columns)
    if missing:
        raise ValueError(f"support sensitivity needs voter issue weights: {sorted(missing)}")

    arms = [("scalar", None)] + [("synthetic_issue_profile", value)
                                 for value in dispersion_list]
    seed_rows, candidate_rows, agent_decision_rows = [], [], []
    for seed in seed_list:
        results = {}
        for mode, dispersion in arms:
            run_config = replace(config, support_mode=mode,
                                 synthetic_issue_dispersion=(
                                     config.synthetic_issue_dispersion if dispersion is None
                                     else dispersion))
            results[(mode, dispersion)] = simulate(
                run_config, data, seed, campaign_actions, agents=agents)
        control = results[("scalar", None)]
        control_shares = control.candidate_results.set_index("candidate_id").vote_share
        control_support = control.final_state.set_index("voter_id")
        control_winner = set(control.turnout["winner_ids"])
        control_decisions = {
            (decision["day"], decision["candidate_id"]): decision
            for decision in control.diagnostics["agent_decisions"]}
        for mode, dispersion in arms:
            result = results[(mode, dispersion)]
            candidate_results = result.candidate_results.set_index("candidate_id")
            voter_state = result.final_state.set_index("voter_id").reindex(control_support.index)
            decisions = {
                (decision["day"], decision["candidate_id"]): decision
                for decision in result.diagnostics["agent_decisions"]}
            decision_keys = set(control_decisions) | set(decisions)
            changed_decisions = sum(
                (None if key not in control_decisions else tuple(
                    control_decisions[key][field] for field in
                    ("message_type", "target_policy", "target_group",
                     "target_candidate_id", "recipients", "taken")))
                != (None if key not in decisions else tuple(
                    decisions[key][field] for field in
                    ("message_type", "target_policy", "target_group",
                     "target_candidate_id", "recipients", "taken")))
                for key in decision_keys)
            common_decisions = set(control_decisions) & set(decisions)
            poll_share_differences = [
                abs(decisions[key]["poll_share"] - control_decisions[key]["poll_share"])
                for key in common_decisions]
            seed_rows.append({
                "seed": seed, "support_mode": mode, "dispersion": dispersion,
                "winner": ";".join(result.turnout["winner_ids"]),
                "winner_changed_vs_scalar": set(result.turnout["winner_ids"]) != control_winner,
                "turnout": result.turnout["turnout"],
                "turnout_rate": result.turnout["turnout_rate"],
                "turnout_rate_difference_vs_scalar": (
                    result.turnout["turnout_rate"] - control.turnout["turnout_rate"]),
                "agent_decision_count": len(result.diagnostics["agent_decisions"]),
                "agent_actions_changed_vs_scalar": changed_decisions,
                "agent_action_change_rate_vs_scalar": (
                    changed_decisions / len(decision_keys) if decision_keys else 0.0),
                "mean_absolute_agent_poll_share_difference_vs_scalar": (
                    float(np.mean(poll_share_differences)) if poll_share_differences else 0.0),
                "mean_absolute_support_difference_vs_scalar": float(np.mean([
                    np.abs(voter_state[f"support_{candidate_id}"].to_numpy()
                           - control_support[f"support_{candidate_id}"].to_numpy()).mean()
                    for candidate_id in control.candidate_results.candidate_id])),
            })
            for decision in result.diagnostics["agent_decisions"]:
                agent_decision_rows.append({
                    "seed": seed, "support_mode": mode, "dispersion": dispersion,
                    **decision,
                })
            for candidate_id in control.candidate_results.candidate_id:
                share = float(candidate_results.vote_share.get(candidate_id, 0.0))
                mean_support = float(voter_state[f"support_{candidate_id}"].mean())
                control_mean_support = float(control_support[f"support_{candidate_id}"].mean())
                control_share = float(control_shares.get(candidate_id, 0.0))
                candidate_rows.append({
                    "seed": seed, "support_mode": mode, "dispersion": dispersion,
                    "candidate_id": candidate_id,
                    "mean_support": mean_support,
                    "mean_support_difference_vs_scalar": mean_support - control_mean_support,
                    "vote_share": share,
                    "vote_share_difference_vs_scalar": share - control_share,
                })

    per_seed = pd.DataFrame(seed_rows)
    per_candidate = pd.DataFrame(candidate_rows)
    summary_rows = []
    for (mode, dispersion), group in per_seed.groupby(
            ["support_mode", "dispersion"], sort=False, dropna=False):
        summary_rows.append({
            "support_mode": mode, "dispersion": dispersion, "seed_count": len(group),
            "winner_changed_rate_vs_scalar": float(group.winner_changed_vs_scalar.mean()),
            "mean_absolute_support_difference_vs_scalar": float(
                group.mean_absolute_support_difference_vs_scalar.mean()),
            "mean_absolute_support_difference_p05": float(
                group.mean_absolute_support_difference_vs_scalar.quantile(0.05)),
            "mean_absolute_support_difference_p95": float(
                group.mean_absolute_support_difference_vs_scalar.quantile(0.95)),
            "mean_turnout_rate_difference_vs_scalar": float(
                group.turnout_rate_difference_vs_scalar.mean()),
            "mean_agent_action_change_rate_vs_scalar": float(
                group.agent_action_change_rate_vs_scalar.mean()),
            "mean_absolute_agent_poll_share_difference_vs_scalar": float(
                group.mean_absolute_agent_poll_share_difference_vs_scalar.mean()),
        })
    candidate_summary_rows = []
    for (mode, dispersion, candidate_id), group in per_candidate.groupby(
            ["support_mode", "dispersion", "candidate_id"], sort=False, dropna=False):
        row = {"support_mode": mode, "dispersion": dispersion,
               "candidate_id": candidate_id, "seed_count": len(group)}
        for metric in ("mean_support_difference_vs_scalar", "vote_share_difference_vs_scalar"):
            values = group[metric]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_p05"] = float(values.quantile(0.05))
            row[f"{metric}_p95"] = float(values.quantile(0.95))
        candidate_summary_rows.append(row)
    metadata = {
        "synthetic": True,
        "claim": "Matched synthetic support-mode sensitivity analysis; not a forecast",
        "base_config": asdict(config), "seeds": seed_list,
        "arms": [{"support_mode": mode, "dispersion": dispersion}
                 for mode, dispersion in arms],
        "control": "Scalar support mode for each same-seed comparison",
        "agents": [asdict(agent) for agent in (agents or [])],
        "agent_decisions_note": (
            "Agent decisions are recorded by seed and support arm. Decisions may differ "
            "because agents respond to polls of each arm's support."),
        "random_stream_note": (
            "Each support arm uses the same seed and shared campaign, poll, event, "
            "turnout, voting, community, and smear-spread random streams."),
        "paired_profile_note": "The same seed and issue-offset draws are used at each dispersion; dispersion scales those draws.",
        "zero_dispersion_note": "Dispersion 0 removes voter-specific offsets but still uses weighted per-issue distance; it is not the scalar-support arm.",
        "issue_weights": "Supplied issue_* fields normalized per voter; they are priorities, not voter policy stances.",
        "synthetic_profile_note": "Issue positions are scalar opinion plus seeded offsets; not measured beliefs.",
        "uncertainty_note": "Percentiles describe only these specified synthetic seeds.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    decision_columns = ["seed", "support_mode", "dispersion", "day", "candidate_id",
                        "strategy", "action_id", "message_type", "target_policy",
                        "target_group", "target_candidate_id", "poll_share", "recipients",
                        "cost", "taken", "budget_left"]
    return SupportSensitivityResult(per_seed, per_candidate,
                                    pd.DataFrame(agent_decision_rows, columns=decision_columns),
                                    pd.DataFrame(summary_rows),
                                    pd.DataFrame(candidate_summary_rows), metadata)


def _constituency_slice(data: dict, constituency_id: int) -> dict:
    """Return one constituency plus its local, regional, and state events."""
    voters = data["voters"].loc[data["voters"].constituency_id.eq(constituency_id)].copy()
    if voters.empty:
        raise ValueError(f"constituency {constituency_id} has no voters")
    candidates = data["candidates"].loc[
        data["candidates"].constituency_id.eq(constituency_id)].copy()
    if candidates.empty:
        raise ValueError(f"constituency {constituency_id} has no candidates")
    voter_ids = set(voters.voter_id)
    edges = data["social_network"]
    edges = edges.loc[edges.source_voter_id.isin(voter_ids)
                      & edges.target_voter_id.isin(voter_ids)].copy()
    events = data.get("events", pd.DataFrame()).copy()
    if not events.empty and {"constituency_id", "region_id"}.issubset(events.columns):
        region_id = voters.region_id.iloc[0]
        local = events.constituency_id.eq(constituency_id)
        broad = events.constituency_id.isna() & (
            events.region_id.isna() | events.region_id.eq(region_id))
        events = events.loc[local | broad].copy()
    return {**data, "voters": voters.reset_index(drop=True),
            "candidates": candidates.reset_index(drop=True),
            "social_network": edges.reset_index(drop=True),
            "events": events.reset_index(drop=True)}


def _validate_seeds(seeds) -> list[int]:
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
           for seed in seed_list):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    return seed_list


def _scenario_items(scenarios):
    if isinstance(scenarios, dict):
        items = [(str(name), spec) for name, spec in scenarios.items()]
    else:
        items = []
        for spec in scenarios:
            if not isinstance(spec, dict) or not spec.get("name"):
                raise ValueError("each scenario must be a mapping with a nonempty name")
            items.append((str(spec["name"]), spec))
    if not items or len({name for name, _ in items}) != len(items):
        raise ValueError("scenarios must be nonempty and have unique names")
    allowed = {"name", "config_overrides", "campaign_actions", "agents",
               "constituency_id", "constituency_ids", "spillover",
               "party_campaigns", "party_agents"}
    for name, spec in items:
        if not isinstance(spec, dict):
            raise ValueError(f"scenario {name!r} must be a mapping")
        unknown = set(spec).difference(allowed)
        if unknown:
            raise ValueError(f"scenario {name!r} has unknown fields: {sorted(unknown)}")
        overrides = spec.get("config_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"scenario {name!r} config_overrides must be a mapping")
        if not set(overrides).issubset(SimulationConfig.__dataclass_fields__):
            raise ValueError(f"scenario {name!r} has unknown SimulationConfig overrides")
        if "constituency_ids" in spec and "constituency_id" in spec:
            raise ValueError(f"scenario {name!r} cannot specify both constituency_id and constituency_ids")
        if "constituency_ids" not in spec and any(
                key in spec for key in ("spillover", "party_campaigns", "party_agents")):
            raise ValueError(f"scenario {name!r} spillover and party-level inputs require constituency_ids")
    return items


def _outcome_rows(result, scenario: str, seed: int, constituency_id: int | None = None):
    """Normalize a single or regional RunResult into candidate-level rows."""
    rows = []
    if hasattr(result, "constituencies"):
        table = result.candidate_results
        turnout_by_id = result.turnout.set_index("constituency_id").to_dict("index")
    else:
        table = result.candidate_results.copy()
        table["constituency_id"] = constituency_id
        turnout_by_id = {constituency_id: result.turnout}
    for cid, constituency in table.groupby("constituency_id", sort=True, dropna=False):
        outcome = turnout_by_id[int(cid)]
        winner_ids = set(outcome.get("winner_ids", []))
        shares = constituency.vote_share.to_numpy(dtype=float)
        support = constituency.get("final_mean_support", constituency.get("mean_support"))
        support_values = (support.to_numpy(dtype=float) if support is not None
                          else np.full(len(constituency), np.nan))
        margin = (float(np.sort(shares)[-1] - np.sort(shares)[-2])
                  if len(shares) > 1 else 1.0)
        support_margin = (float(np.sort(support_values)[-1] - np.sort(support_values)[-2])
                          if len(support_values) > 1 else 1.0)
        for record in constituency.to_dict("records"):
            rows.append({
                "scenario": scenario, "seed": seed, "constituency_id": int(cid),
                "candidate_id": record["candidate_id"],
                "party_id": record.get("party_id"),
                "mean_support": float(record.get("mean_support",
                                                  record.get("final_mean_support", np.nan))),
                "votes": int(record.get("votes", 0)),
                "vote_share": float(record.get("vote_share", 0.0)),
                "turnout": int(outcome.get("turnout", 0)),
                "turnout_rate": float(outcome.get("turnout_rate", 0.0)),
                "winner": record["candidate_id"] in winner_ids,
                "top_two_vote_share_margin": margin,
                "top_two_support_margin": support_margin,
            })
    return rows


def run_batch(config: SimulationConfig, data: dict, scenarios, seeds) -> pd.DataFrame:
    """Run named scenarios over matched seeds and return tidy candidate outcomes.

    A scenario is a mapping with optional ``config_overrides``, scripted
    ``campaign_actions``, ``agents`` and a ``constituency_id``. For regional
    scenarios, provide ``constituency_ids`` plus optional ``spillover``,
    ``party_campaigns`` and ``party_agents``. Actions and agents must be
    ``CampaignAction``/``CandidateAgent`` instances. Each output row is one
    (scenario, seed, constituency, candidate) outcome.
    """
    seed_list = _validate_seeds(seeds)
    scenario_list = _scenario_items(scenarios)
    rows = []
    for name, spec in scenario_list:
        run_config = replace(config, **spec.get("config_overrides", {}))
        actions = list(spec.get("campaign_actions", ()))
        agents = list(spec.get("agents", ()))
        if any(not isinstance(action, CampaignAction) for action in actions):
            raise TypeError(f"scenario {name!r} campaign_actions must contain CampaignAction values")
        if any(not isinstance(agent, CandidateAgent) for agent in agents):
            raise TypeError(f"scenario {name!r} agents must contain CandidateAgent values")
        constituency_ids = spec.get("constituency_ids")
        if constituency_ids is not None:
            from .region import simulate_region
            for seed in seed_list:
                result = simulate_region(
                    run_config, constituency_ids, seed=seed,
                    spillover=spec.get("spillover", 0.0), data=data,
                    campaign_actions=actions, agents=agents,
                    party_campaigns=spec.get("party_campaigns"),
                    party_agents=spec.get("party_agents"))
                rows.extend(_outcome_rows(result, name, seed))
        else:
            cid = spec.get("constituency_id", run_config.constituency_id)
            local_data = _constituency_slice(data, cid)
            run_config = replace(run_config, constituency_id=cid)
            for seed in seed_list:
                result = simulate(run_config, local_data, seed,
                                  campaign_actions=actions, agents=agents)
                rows.extend(_outcome_rows(result, name, seed, cid))
    frame = pd.DataFrame(rows)
    frame.attrs["synthetic"] = True
    frame.attrs["seed_count"] = len(seed_list)
    frame.attrs["seed_list"] = seed_list
    frame.attrs["scenario_names"] = [name for name, _ in scenario_list]
    return frame


def summarize_batch(per_run: pd.DataFrame) -> pd.DataFrame:
    """Summarize batch outcomes with median and empirical 5th–95th percentiles."""
    required = {"scenario", "seed", "constituency_id", "candidate_id",
                "mean_support", "vote_share", "turnout_rate", "winner",
                "top_two_vote_share_margin", "top_two_support_margin"}
    missing = required.difference(per_run.columns)
    if missing:
        raise ValueError(f"batch outcomes are missing columns: {sorted(missing)}")
    rows = []
    keys = ["scenario", "constituency_id", "candidate_id"]
    for key, group in per_run.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["seed_count"] = int(group.seed.nunique())
        row["win_rate"] = float(group.winner.mean())
        for metric in ("mean_support", "vote_share", "turnout_rate",
                       "top_two_vote_share_margin", "top_two_support_margin"):
            values = group[metric].astype(float)
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_p05"] = float(values.quantile(.05))
            row[f"{metric}_p95"] = float(values.quantile(.95))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_campaign(config: SimulationConfig, data: dict, seeds,
                     control_actions=None, treatment_actions=None,
                     control_agents=None, treatment_agents=None,
                     constituency_id: int | None = None) -> pd.DataFrame:
    """Compare matched campaign/control runs; all non-campaign settings match.

    ``control_actions`` and ``treatment_actions`` may differ, as may candidate
    agent lists. The config, data, and seed streams are identical. The output
    has one row per seed, constituency and candidate, with control/treatment
    outcomes and treatment-minus-control differences.
    """
    seed_list = _validate_seeds(seeds)
    cid = config.constituency_id if constituency_id is None else constituency_id
    local_data = _constituency_slice(data, cid)
    local_config = replace(config, constituency_id=cid)
    batch = run_batch(local_config, local_data, {
        "control": {"campaign_actions": list(control_actions or ()),
                    "agents": list(control_agents or ()), "constituency_id": cid},
        "treatment": {"campaign_actions": list(treatment_actions or ()),
                      "agents": list(treatment_agents or ()), "constituency_id": cid},
    }, seed_list)
    keys = ["seed", "constituency_id", "candidate_id"]
    metrics = ["mean_support", "votes", "vote_share", "turnout", "turnout_rate",
               "winner", "top_two_vote_share_margin"]
    control = batch.loc[batch.scenario.eq("control"), keys + metrics].rename(
        columns={metric: f"{metric}_control" for metric in metrics})
    treatment = batch.loc[batch.scenario.eq("treatment"), keys + metrics].rename(
        columns={metric: f"{metric}_treatment" for metric in metrics})
    paired = control.merge(treatment, on=keys, validate="one_to_one")
    for metric in ("mean_support", "votes", "vote_share", "turnout", "turnout_rate",
                   "top_two_vote_share_margin"):
        paired[f"{metric}_difference"] = paired[f"{metric}_treatment"] - paired[f"{metric}_control"]
    paired["candidate_winner_status_changed"] = (
        paired.winner_control != paired.winner_treatment)
    race_changes = {}
    for key, group in paired.groupby(["seed", "constituency_id"], sort=False):
        control_winners = set(group.loc[group.winner_control, "candidate_id"])
        treatment_winners = set(group.loc[group.winner_treatment, "candidate_id"])
        race_changes[key] = control_winners != treatment_winners
    paired["constituency_winner_changed"] = [
        race_changes[(row.seed, row.constituency_id)]
        for row in paired[["seed", "constituency_id"]].itertuples(index=False)]
    paired.attrs["synthetic"] = True
    paired.attrs["seed_list"] = seed_list
    paired.attrs["comparison"] = "Matched seeds; only campaign actions/agents differ"
    return paired


def build_step5_scenarios(config: SimulationConfig, data: dict) -> dict:
    """Build the Step 5 scenario matrix with matched action contact budgets.

    The returned arms cover frozen/social controls, ordinary/misleading/smear
    campaigns, news, the combined model, scripted versus three agent
    strategies, and equal-reach targeting policies. Scenario parameters are
    illustrative synthetic controls, not fitted campaign budgets.
    """
    voters = data["voters"]
    candidates = data["candidates"].sort_values("candidate_id").candidate_id.tolist()
    if len(candidates) < 2:
        raise ValueError("the Step 5 matrix needs at least two candidates")
    days = config.days
    lead, target = candidates[0], candidates[1]
    population = len(voters)
    reach = 0.05
    intensity = 0.5
    duration = min(6, days)
    start = min(10, max(1, days - duration + 1))
    active_campaign = max(config.lambda_campaign, 0.2)
    base_actions = [
        CampaignAction("matrix_ordinary", lead, start, duration, intensity, reach),
        CampaignAction("matrix_misleading", lead, start, duration, intensity, reach,
                       message_type="misleading", distortion=0.5),
        CampaignAction("matrix_smear", lead, start, duration, intensity, reach,
                       message_type="smear", target_candidate_id=target),
    ]
    group = "Students" if ("occupation" in voters and voters.occupation.eq("Student").any()) else None
    policy_specs = [("uniform", "uniform", None),
                    ("high_susceptibility", "high_susceptibility", None),
                    ("high_influence", "high_influence", None),
                    ("community", "community", None)]
    if group is not None:
        policy_specs.insert(1, ("group_students", "group", group))
    targeting = {
        f"target_{name}": {"config_overrides": {"lambda_campaign": active_campaign,
                                                   "lambda_event": 0.0},
                           "campaign_actions": [CampaignAction(
                               "equal_budget_targeting", lead, start, duration, intensity,
                               reach, target_policy=policy, target_group=target_group)]}
        for name, policy, target_group in policy_specs
    }
    scripted = [CampaignAction(f"scripted_{candidate}", candidate, 1, days,
                               intensity, reach) for candidate in candidates]
    contacts_per_agent = int(round(population * reach)) * days
    agent_arms = {}
    for strategy in ("greedy", "go_negative", "rebut"):
        agent_arms[f"agents_{strategy}"] = {
            "config_overrides": {"lambda_campaign": active_campaign,
                                 "lambda_event": 0.0},
            "agents": [CandidateAgent(candidate, strategy=strategy,
                                      budget=contacts_per_agent, start_day=1,
                                      decide_every=3, reach_fraction=reach,
                                      intensity=intensity)
                       for candidate in candidates],
        }
    scenarios = {
        "frozen": {"config_overrides": {"lambda_social": 0.0,
                                           "lambda_campaign": 0.0,
                                           "lambda_event": 0.0}},
        "social_only": {"config_overrides": {"lambda_campaign": 0.0,
                                                "lambda_event": 0.0}},
        "campaign_ordinary": {"config_overrides": {"lambda_campaign": active_campaign,
                                                      "lambda_event": 0.0},
                              "campaign_actions": [base_actions[0]]},
        "campaign_misleading": {"config_overrides": {"lambda_campaign": active_campaign,
                                                        "lambda_event": 0.0},
                                "campaign_actions": [base_actions[1]]},
        "campaign_smear": {"config_overrides": {"lambda_campaign": active_campaign,
                                                  "lambda_event": 0.0},
                           "campaign_actions": [base_actions[2]]},
        "social_plus_news": {"config_overrides": {"lambda_campaign": 0.0}},
        "all_together": {"config_overrides": {"lambda_campaign": active_campaign},
                         "campaign_actions": base_actions},
        "scripted_equal_budget": {"config_overrides": {"lambda_campaign": active_campaign,
                                                          "lambda_event": 0.0},
                                  "campaign_actions": scripted},
        **agent_arms,
        **targeting,
    }
    return scenarios
