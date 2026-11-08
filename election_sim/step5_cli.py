"""Run the roadmap Step 5 baseline scan, scenario matrix, and spillover sweep."""

import argparse
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .experiments import (build_step5_scenarios, run_batch, summarize_batch,
                          _constituency_slice)
from .region import load_region


def run_step5(config: SimulationConfig, data: dict, seed: int = 0,
              seed_count: int = 30) -> dict:
    """Execute Step 5 analyses and return tables plus a reproducibility manifest."""
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ValueError("seed_count must be a positive integer")
    constituency_ids = sorted(data["voters"].constituency_id.unique().tolist())
    baseline_scenarios = {
        f"C{int(cid):02d}": {"constituency_id": int(cid),
                             "config_overrides": {"lambda_campaign": 0.0}}
        for cid in constituency_ids
    }
    baseline = run_batch(config, data, baseline_scenarios, [seed])
    gaps = (baseline.groupby("constituency_id", sort=True)
            .top_two_support_margin.first().sort_values())
    closest = int(gaps.index[0])
    close_race = bool(gaps.iloc[0] <= 0.10)

    local_data = _constituency_slice(data, closest)
    local_config = replace(config, constituency_id=closest)
    scenarios = build_step5_scenarios(local_config, local_data)
    seeds = list(range(seed, seed + seed_count))
    per_seed = run_batch(local_config, local_data, scenarios, seeds)
    summary = summarize_batch(per_seed)

    paired_specs = {
        "campaign_ordinary": "social_only",
        "campaign_misleading": "social_only",
        "campaign_smear": "social_only",
        "agents_greedy": "scripted_equal_budget",
        "agents_go_negative": "scripted_equal_budget",
        "agents_rebut": "scripted_equal_budget",
        "target_group_students": "target_uniform",
        "target_high_susceptibility": "target_uniform",
        "target_high_influence": "target_uniform",
        "target_community": "target_uniform",
    }
    paired_rows = []
    index_keys = ["seed", "constituency_id", "candidate_id"]
    compared_metrics = ["mean_support", "votes", "vote_share", "turnout_rate",
                        "winner", "top_two_vote_share_margin",
                        "top_two_support_margin"]
    for treatment_name, control_name in paired_specs.items():
        if treatment_name not in scenarios or control_name not in scenarios:
            continue
        control = per_seed.loc[per_seed.scenario.eq(control_name),
                               index_keys + compared_metrics].rename(
            columns={metric: f"{metric}_control" for metric in compared_metrics})
        treatment = per_seed.loc[per_seed.scenario.eq(treatment_name),
                                 index_keys + compared_metrics].rename(
            columns={metric: f"{metric}_treatment" for metric in compared_metrics})
        paired = control.merge(treatment, on=index_keys, validate="one_to_one")
        paired["comparison"] = treatment_name
        for metric in ("mean_support", "votes", "vote_share", "turnout_rate",
                       "top_two_vote_share_margin", "top_two_support_margin"):
            paired[f"{metric}_difference"] = (
                paired[f"{metric}_treatment"] - paired[f"{metric}_control"])
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
        paired_rows.append(paired)
    paired_effects = pd.concat(paired_rows, ignore_index=True)

    region_id = int(local_data["voters"].region_id.iloc[0])
    region_ids = sorted(data["voters"].loc[data["voters"].region_id.eq(region_id),
                                           "constituency_id"].unique().astype(int).tolist())
    spillover_scenarios = {
        f"spillover_{value:g}": {"constituency_ids": region_ids, "spillover": value}
        for value in (0.0, 0.05, 0.1, 0.2)
    }
    spillover_per_seed = run_batch(config, data, spillover_scenarios, seeds)
    spillover_summary = summarize_batch(spillover_per_seed)
    def hash_table(table, keys):
        ordered = table.sort_values([key for key in keys if key in table.columns],
                                    kind="mergesort") if any(key in table.columns for key in keys) else table
        return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()

    def serializable_scenario(spec):
        return {key: ([asdict(value) if is_dataclass(value) else value for value in values]
                      if key in {"campaign_actions", "agents"} else values)
                for key, values in spec.items()}

    manifest = {
        "synthetic": True,
        "claim": "Scenario sensitivity workflow; not an empirical election forecast",
        "base_config": asdict(config),
        "scenario_definitions": {name: serializable_scenario(spec)
                                 for name, spec in scenarios.items()},
        "data_sha256": {
            "voters": hash_table(data["voters"], ["voter_id"]),
            "candidates": hash_table(data["candidates"], ["candidate_id"]),
            "social_network": hash_table(data["social_network"],
                                         ["target_voter_id", "source_voter_id"]),
            "events": hash_table(data.get("events", pd.DataFrame()), ["event_id"]),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_scan_seed": seed,
        "baseline_scan_constituencies": constituency_ids,
        "baseline_scan_metric": "smallest top-two final mean-support gap",
        "selected_constituency_id": closest,
        "selected_top_two_support_margin": float(gaps.iloc[0]),
        "selected_within_10_percentage_points": close_race,
        "seed_list": seeds,
        "scenario_names": list(scenarios),
        "spillover_values": [0.0, 0.05, 0.1, 0.2],
        "spillover_constituencies": region_ids,
        "uncertainty_note": "Seed intervals describe only these synthetic scenario runs.",
    }
    return {"baseline_scan": baseline, "per_seed": per_seed,
            "summary": summary, "paired_effects": paired_effects,
            "spillover_per_seed": spillover_per_seed,
            "spillover_summary": spillover_summary, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/synthetic_election_dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing paths are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_region(args.data_dir, days=config.days)
        result = run_step5(config, data, args.seed, args.seed_count)
        args.output.mkdir(parents=True, exist_ok=False)
        for key in ("baseline_scan", "per_seed", "summary", "paired_effects",
                    "spillover_per_seed", "spillover_summary"):
            result[key].to_csv(args.output / f"{key}.csv", index=False)
        (args.output / "manifest.json").write_text(
            json.dumps(result["manifest"], indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print("SYNTHETIC Step 5 sweep; intervals describe these seeds only.")
    print(f"Closest baseline constituency: C{result['manifest']['selected_constituency_id']:02d}; "
          f"top-two support gap={result['manifest']['selected_top_two_support_margin']:.3f}")
    print(result["summary"].to_string(index=False))
    print(f"Saved Step 5 tables to {args.output.resolve()}")


if __name__ == "__main__":
    main()
