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
    parser.add_argument("--output", type=Path, required=True,
                        help="Run directory (must not exist unless --overwrite is given)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output directory if it already holds a previous run")
    parser.add_argument("--record-history", action="store_true",
                        help="Also save every voter's daily state to history.npz and history_timeline.json")
    args = parser.parse_args()
    try:
        if args.output.exists():
            if not args.overwrite:
                raise ValueError(f"output already exists: {args.output}; "
                                 "choose a new run directory or add --overwrite")
            # Only ever delete a folder that holds a previous run.
            if not (args.output / "manifest.json").exists():
                raise ValueError(f"{args.output} is not a previous run folder (no manifest.json); "
                                 "refusing to overwrite it")
            shutil.rmtree(args.output)
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_constituency(args.data_dir, config.constituency_id,
                                 election_duration_days=config.days, events_file=args.events_file)
        candidate_ids = set(data["candidates"].candidate_id)
        actions = (load_actions(args.campaign_actions, config.days, candidate_ids)
                   if args.campaign_actions else None)
        agents = load_agents(args.agents, candidate_ids) if args.agents else None
        result = simulate(config, data, args.seed, actions, record_history=args.record_history,
                          agents=agents)
        event_path = args.events_file or args.data_dir / "political_events.csv"
        result.metadata["data_file_sha256"] = {
            **{name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
               for name in ("voters.csv", "social_network.csv", "candidates.csv")},
            "political_events.csv": hashlib.sha256(event_path.read_bytes()).hexdigest()}
        result.metadata["event_schedule_path"] = str(event_path.resolve())
        result.save(args.output)
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
