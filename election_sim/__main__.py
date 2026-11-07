"""Run with python -m election_sim from the repository root."""

import argparse
import hashlib
import json
from pathlib import Path

from constituency_loader import load_constituency

from .config import SimulationConfig
from .campaign import load_actions
from .simulation import simulate


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic social simulation with ordinary/smear campaigns, targeting, and scheduled event effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("From the repository root, install pinned dependencies:\n"
                "  python -m pip install -r requirements.txt\n\n"
                "Run either control (choose a new output directory each time):\n"
                "  python -m election_sim --config configs/baseline.json --seed 42 --output results/social_seed42\n"
                "  python -m election_sim --config configs/frozen.json --seed 42 --output results/frozen_seed42\n\n"
                "Run an ordinary or smear campaign:\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_c01.json --seed 42 --output results/ordinary_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/smear_c01.json --seed 42 --output results/smear_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_uniform_5pct_c01.json --seed 42 --output results/uniform_5pct_seed42\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/ordinary_students_c01.json --seed 42 --output results/students_seed42\n"
                "  python -m election_sim --config configs/baseline.json --events-file event_scenarios/c01_moderate.csv --seed 42 --output results/c01_moderate_events_seed42\n\n"
                "Verify the baseline and loader:\n"
                '  python -m unittest discover -s tests -p "test_*.py" -v\n'
                '  python -m unittest discover -s constituency_loader -p "test_*.py" -v'))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_election_dataset"))
    parser.add_argument("--events-file", type=Path,
                        help="Optional alternate event schedule CSV (default: political_events.csv from data-dir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--campaign-actions", type=Path,
                        help="JSON campaign actions with uniform, group, high_susceptibility, or high_influence targeting")
    parser.add_argument("--output", type=Path, required=True, help="New run directory (must not exist)")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}; choose a new run directory")
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_constituency(args.data_dir, config.constituency_id,
                                 election_duration_days=config.days, events_file=args.events_file)
        actions = (load_actions(args.campaign_actions, config.days, set(data["candidates"].candidate_id))
                   if args.campaign_actions else None)
        result = simulate(config, data, args.seed, actions)
        event_path = args.events_file or args.data_dir / "political_events.csv"
        result.metadata["data_file_sha256"] = {
            **{name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
               for name in ("voters.csv", "social_network.csv", "candidates.csv")},
            "political_events.csv": hashlib.sha256(event_path.read_bytes()).hexdigest()}
        result.metadata["event_schedule_path"] = str(event_path.resolve())
        result.save(args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Baseline failed: {exc}\n")
    print("SYNTHETIC ELECTION: no real election forecast; scheduled event effects are model assumptions.")
    print(result.candidate_results.to_string(index=False))
    print(json.dumps(result.turnout, indent=2))
    print(f"Saved run to {args.output.resolve()}")


if __name__ == "__main__":
    main()
