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
        description="Synthetic social simulation with optional ordinary campaigns; events disabled",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("From the repository root, install pinned dependencies:\n"
                "  python -m pip install -r requirements.txt\n\n"
                "Run either control (choose a new output directory each time):\n"
                "  python -m election_sim --config configs/baseline.json --seed 42 --output results/social_seed42\n"
                "  python -m election_sim --config configs/frozen.json --seed 42 --output results/frozen_seed42\n\n"
                "Run initial ordinary campaign:\n"
                "  python -m election_sim --config configs/campaign.json --campaign-actions data/campaign_actions/ordinary_c01.json --seed 42 --output results/campaign_seed42\n\n"
                "Verify the baseline and loader:\n"
                '  python -m unittest discover -s tests -p "test_*.py" -v\n'
                '  python -m unittest discover -s constituency_loader -p "test_*.py" -v'))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_election_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--campaign-actions", type=Path,
                        help="JSON campaign actions; initial implementation supports ordinary uniform actions")
    parser.add_argument("--output", type=Path, required=True, help="New run directory (must not exist)")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}; choose a new run directory")
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        data = load_constituency(args.data_dir, config.constituency_id)
        actions = (load_actions(args.campaign_actions, config.days, set(data["candidates"].candidate_id))
                   if args.campaign_actions else None)
        result = simulate(config, data, args.seed, actions)
        result.metadata["data_file_sha256"] = {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ("voters.csv", "social_network.csv", "candidates.csv", "political_events.csv")}
        result.save(args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Baseline failed: {exc}\n")
    print("SYNTHETIC ELECTION: no real election forecast; events are disabled.")
    print(result.candidate_results.to_string(index=False))
    print(json.dumps(result.turnout, indent=2))
    print(f"Saved run to {args.output.resolve()}")


if __name__ == "__main__":
    main()
