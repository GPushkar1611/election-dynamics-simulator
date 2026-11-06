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
from .campaign.effects import action_rng, apply_ordinary, apply_smear, select_recipients
from .campaign.models import CampaignAction, check_actions
from .dynamics import social_step
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


def _scenario(config: SimulationConfig, actions: list[CampaignAction]) -> str:
    types = {action.message_type for action in actions}
    if "smear" in types:
        return "smear_campaign"
    if "ordinary" in types:
        return "ordinary_campaign"
    return "frozen" if config.lambda_social == 0 else "social_only"


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
    children = np.random.SeedSequence(seed).spawn(len(names))
    child_seeds = {name: [int(v) for v in child.generate_state(4)]
                   for name, child in zip(names, children)}
    rng = {name: np.random.default_rng(value) for name, value in child_seeds.items()}
    candidate_ids = candidates.candidate_id.tolist()
    candidate_positions = dict(zip(candidate_ids, candidates[[
        "position_employment", "position_economy", "position_education",
        "position_healthcare", "position_agriculture"]].mean(axis=1)))
    recipients = {
        action.action_id: select_recipients(action, len(voters),
                                            action_rng(child_seeds["campaign"], action.action_id))
        for action in actions
    }
    # Smear penalties lower utility per voter and candidate; zero means no smear.
    penalty = np.zeros((len(voters), len(candidate_ids)))
    summaries, support_rows, distributions, sampled_rows = [], [], [], []
    # A deterministic systematic ID sample for view state, not an electorate estimate.
    sample_indices = np.linspace(0, len(voters) - 1, min(100, len(voters)), dtype=int)
    bins = np.linspace(-1, 1, 21)
    support = candidate_support(opinions, candidates, config.b_pop, config.b_ideol, config.tau, penalty)
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
            support = candidate_support(opinions, candidates, config.b_pop, config.b_ideol, config.tau, penalty)
        # Counts reflect the state logged for this day; the next update uses it.
        eligible = int((np.abs(opinions[graph.sources] - opinions[graph.targets]) <=
                        confidence[graph.targets]).sum())
        quantiles = np.quantile(opinions, [0.05, 0.25, 0.5, 0.75, 0.95])
        active_actions = [action for action in actions if action.active_on(day)]
        summaries.append({"day": day, "opinion_mean": float(opinions.mean()),
                          "opinion_sd": float(opinions.std()),
                          "polarization_variance": float(opinions.var()),
                          **dict(zip(("q05", "q25", "q50", "q75", "q95"), map(float, quantiles))),
                          "eligible_edges": eligible,
                          "active_campaign_actions": len(active_actions),
                          "campaign_exposures_today": int(sum(len(recipients[action.action_id])
                                                               for action in active_actions)),
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
                   "events_applied": 0,
                   "loaded_events_not_applied": len(data.get("events", []))}
    metadata = {"synthetic": True, "claim": "Synthetic scenario; not an empirically validated forecast",
                "scenario": _scenario(config, actions),
                "config": asdict(config), "master_seed": seed, "child_seeds": child_seeds,
                "campaign_actions": [asdict(action) for action in actions],
                "rng": "numpy.default_rng (PCG64); canonical voter/candidate ID order",
                "view_sample": "Up to 100 systematically spaced canonical voter IDs; internal visualization only, not a representative sample",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "selected_table_sha256": {"voters": _fingerprint(voters), "candidates": _fingerprint(candidates),
                                          "social_network": _fingerprint(edges)},
                "code": _code_revision()}
    return RunResult(pd.DataFrame(summaries), pd.DataFrame(support_rows), pd.DataFrame(distributions),
                     results, state, pd.DataFrame(sampled_rows), turnout, diagnostics, metadata)
