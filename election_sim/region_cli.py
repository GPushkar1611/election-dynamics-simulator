"""Regional command-line entry point: python -m election_sim.region_cli."""
import argparse
import json
from pathlib import Path
from .config import SimulationConfig
from .region import load_region, simulate_region, spillover_sweep


def main():
    parser = argparse.ArgumentParser(description="Synthetic regional election with social spillover")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic_election_dataset"))
    parser.add_argument("--constituencies", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spillover", type=float, default=0.)
    parser.add_argument("--events-file", type=Path)
    parser.add_argument("--party-scenario", type=Path, help="JSON with party_campaigns and/or party_agents lists")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Also save paired 0, .05, .1, .2 spillover results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("output directory already exists; choose a new directory")
        config = SimulationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        scenario = json.loads(args.party_scenario.read_text(encoding="utf-8")) if args.party_scenario else {}
        if not isinstance(scenario, dict) or set(scenario) - {"party_campaigns", "party_agents"}:
            raise ValueError("party scenario accepts only party_campaigns and party_agents")
        data = load_region(args.data_dir, config.days, args.events_file)
        result = simulate_region(config, args.constituencies, args.seed, args.spillover,
            data=data, record_history=args.record_history, **scenario)
        sweep = spillover_sweep(config, args.constituencies, args.seed, data=data, **scenario) if args.sweep else None
        result.save(args.output)
        if sweep is not None:
            sweep.to_csv(args.output / "spillover_sweep.csv", index=False)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Run failed: {exc}\n")
    print("SYNTHETIC REGIONAL ELECTION: scenario assumptions, not a real election forecast.")
    print(result.seats.to_string(index=False))
    print(f"Saved run to {args.output.resolve()}")


if __name__ == "__main__":
    main()
