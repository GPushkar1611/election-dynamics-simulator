"""Run with python -m election_sim from the repository root."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from constituency_loader import load_constituency

import pandas as pd

from .agents import load_agents
from .campaign import load_actions
from .config import SimulationConfig
from .experiments import (run_event_factorial_sweep, run_event_seed_sweep,
                          run_support_sensitivity_sweep)
from .simulation import simulate


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic election simulation with social influence, campaigns, candidate agents and events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("From the repository root, install pinned dependencies:\n"
                "  python -m pip install -r requirements.txt\n\n"
                "Run either control (use a new output directory, or add --overwrite to replace one):\n"
                "  python -m election_sim --config configs/baseline.json --seed 42 --output results/social_seed42\n"
                "  python -m election_sim --config configs/frozen.json --seed 42 --output results/frozen_seed42\n\n"
                "Run an ordinary or smear campaign:\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_c01.json --seed 42 --output results/ordinary_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/smear_c01.json --seed 42 --output results/smear_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_uniform_5pct_c01.json --seed 42 --output results/uniform_5pct_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_students_c01.json --seed 42 --output results/students_seed42\n"
                "  python -m election_sim --config configs/baseline.json --events-file event_scenarios/c01_moderate.csv --seed 42 --output results/c01_moderate_events_seed42\n\n"
                "Let candidate agents run the campaign (they poll and decide each few days):\n"
                "  python -m election_sim --config configs/agents.json --agents campaign_agents/c01_agents.json --seed 42 --output results/agents_seed42\n\n"
                "Save full per-day voter history (for animations):\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/smear_c01.json --seed 42 --record-history --output results/smear_history_seed42\n\n"
                "Run paired event experiments:\n"
                "  python -m election_sim --config configs/baseline.json --events-file event_scenarios/c01_moderate.csv --event-seed-count 30 --seed 0 --output results/c01_event_sweep\n"
                "  python -m election_sim --config configs/baseline.json --events-file event_scenarios/c01_moderate.csv --event-factorial --event-seed-count 30 --seed 0 --output results/c01_event_factorial\n\n"
                "Compare scalar and synthetic issue-aware support:\n"
                "  python -m election_sim --config configs/baseline.json --support-sensitivity --support-seed-count 30 --support-dispersions 0 0.2 0.35 0.6 1 --seed 0 --output results/c01_support_sensitivity\n"
                "  python -m election_sim --config configs/agents.json --agents campaign_agents/c01_agents.json --support-sensitivity --support-seed-count 30 --support-dispersions 0 0.2 0.35 0.6 1 --seed 0 --output results/c01_agent_support_sensitivity\n\n"
                "Verify the baseline and loader:\n"
                '  python -m unittest discover -s tests -p "test_*.py" -v\n'
                '  python -m unittest discover -s constituency_loader -p "test_*.py" -v'))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_election_dataset"))
    parser.add_argument("--events-file", type=Path,
                        help="Optional alternate event schedule CSV (default: political_events.csv from data-dir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--campaign-actions", type=Path,
                        help="JSON scripted campaign actions (see campaign_actions/README.md)")
    parser.add_argument("--agents", type=Path,
                        help="JSON candidate agents that choose actions during the run (see campaign_agents/README.md)")
    parser.add_argument("--event-seed-count", type=int,
                        help="Run paired shock/no-shock experiments for this many consecutive seeds")
    parser.add_argument("--event-factorial", action="store_true",
                        help="Compare separated event arms; requires --event-seed-count")
    parser.add_argument("--support-sensitivity", action="store_true",
                        help="Compare scalar and synthetic issue-aware support across matched seeds")
    parser.add_argument("--support-seed-count", type=int,
                        help="Number of matched seeds for --support-sensitivity")
    parser.add_argument("--support-dispersions", nargs="+", type=float,
                        default=[0.0, 0.2, 0.35, 0.6, 1.0],
                        help="Synthetic issue-profile dispersions in [0, 1]")
    parser.add_argument("--output", type=Path, required=True,
                        help="Run directory (must not exist unless --overwrite is given)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output directory if it already holds a previous run")
    parser.add_argument("--record-history", action="store_true",
                        help="Also save every voter's daily state to history.npz and history_timeline.json")
    args = parser.parse_args()

    def save_result(result):
        if args.output.exists():
            if not args.overwrite:
                raise ValueError(f"output already exists: {args.output}; "
                                 "choose a new run directory or add --overwrite")
            if not (args.output / "manifest.json").exists():
                raise ValueError(f"{args.output} is not a previous run folder (no manifest.json); "
                                 "refusing to overwrite it")
            shutil.rmtree(args.output)
        result.save(args.output)

    try:
        if args.output.exists():
            if not args.overwrite:
                raise ValueError(f"output already exists: {args.output}; "
                                 "choose a new run directory or add --overwrite")
            # Only ever delete a folder that holds a previous run.
            if not (args.output / "manifest.json").exists():
                raise ValueError(f"{args.output} is not a previous run folder (no manifest.json); "
                                 "refusing to overwrite it")
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_constituency(args.data_dir, config.constituency_id,
                                 election_duration_days=config.days, events_file=args.events_file)
        candidate_ids = set(data["candidates"].candidate_id)
        actions = (load_actions(args.campaign_actions, config.days, candidate_ids)
                   if args.campaign_actions else None)
        agents = load_agents(args.agents, candidate_ids) if args.agents else None
        if args.event_factorial and args.event_seed_count is None:
            raise ValueError("--event-factorial requires --event-seed-count")
        if args.support_sensitivity and args.support_seed_count is None:
            raise ValueError("--support-sensitivity requires --support-seed-count")
        if args.support_sensitivity and args.event_seed_count is not None:
            raise ValueError("support sensitivity and event sweep modes must be run separately")
        if args.support_seed_count is not None and not args.support_sensitivity:
            raise ValueError("--support-seed-count requires --support-sensitivity")
        if args.event_seed_count is not None and (agents or args.record_history):
            raise ValueError("event sweeps do not accept candidate agents or --record-history")
        if args.support_sensitivity and args.record_history:
            raise ValueError("support sensitivity does not accept --record-history")
        event_path = args.events_file or args.data_dir / "political_events.csv"
        data_hashes = {
            **{name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
               for name in ("voters.csv", "social_network.csv", "candidates.csv")},
            "political_events.csv": hashlib.sha256(event_path.read_bytes()).hexdigest()}
        if args.event_seed_count is not None:
            if args.event_seed_count < 1:
                raise ValueError("event_seed_count must be positive")
            seeds = range(args.seed, args.seed + args.event_seed_count)
            result = (run_event_factorial_sweep(config, data, seeds, actions)
                      if args.event_factorial else
                      run_event_seed_sweep(config, data, seeds, actions))
            result.metadata["data_file_sha256"] = data_hashes
            result.metadata["event_schedule_path"] = str(event_path.resolve())
            save_result(result)
            print("SYNTHETIC EVENT SWEEP: intervals summarize these seeds only.")
            if args.event_factorial:
                print(result.arm_summary.to_string(index=False))
            else:
                print(result.overall_summary.to_string(index=False))
            print(result.candidate_summary.to_string(index=False))
            print(f"Saved paired sweep to {args.output.resolve()}")
            return
        if args.support_sensitivity:
            if args.support_seed_count < 1:
                raise ValueError("support_seed_count must be positive")
            seeds = range(args.seed, args.seed + args.support_seed_count)
            result = run_support_sensitivity_sweep(
                config, data, seeds, args.support_dispersions, actions, agents=agents)
            result.metadata["data_file_sha256"] = data_hashes
            result.metadata["event_schedule_path"] = str(event_path.resolve())
            save_result(result)
            print("SYNTHETIC SUPPORT SENSITIVITY: summary describes these seeds and assumptions only.")
            print(result.summary.to_string(index=False))
            print(result.candidate_summary.to_string(index=False))
            print(f"Saved support sensitivity to {args.output.resolve()}")
            return
        result = simulate(config, data, args.seed, actions, record_history=args.record_history,
                          agents=agents)
        result.metadata["data_file_sha256"] = data_hashes
        result.metadata["event_schedule_path"] = str(event_path.resolve())
        save_result(result)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Run failed: {exc}\n")
    print("SYNTHETIC ELECTION: no real election forecast; scheduled event effects are model assumptions.")
    decisions = result.diagnostics["agent_decisions"]
    if decisions:
        print("\nAgent decisions:")
        print(pd.DataFrame(decisions)[["day", "candidate_id", "strategy", "message_type", "target_policy",
                                       "target_group", "target_candidate_id", "poll_share", "recipients",
                                       "taken", "budget_left"]].round(3).to_string(index=False))
    if any(result.diagnostics["smear_believers"].values()):
        print("\nVoters believing a smear about each candidate:", result.diagnostics["smear_believers"])
    print()
    print(result.candidate_results.to_string(index=False))
    print(json.dumps(result.turnout, indent=2))
    print(f"Saved run to {args.output.resolve()}")


if __name__ == "__main__":
    main()
