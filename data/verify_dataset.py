"""Plan section 2.1: regenerate defaults in a temporary folder and record hashes.

Run from the repository root: python data/verify_dataset.py
The supplied CSVs are read-only; data/manifest.json records the comparison.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dataset_generator import Config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    supplied = root / "data" / "synthetic_election_dataset"
    generator = root / "data" / "dataset_generator.py"
    config = Config()
    entries = {}
    with TemporaryDirectory(prefix="election-data-verification-") as temporary:
        command = [sys.executable, str(generator), "--seed", str(config.seed), "--output-dir", temporary]
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        for name in ("voters.csv", "social_network.csv", "candidates.csv", "political_events.csv"):
            original = supplied / name
            regenerated = Path(temporary) / name
            first, second = sha256(original), sha256(regenerated)
            entries[name] = {"supplied_sha256": first, "regenerated_sha256": second,
                             "row_count": len(pd.read_csv(original)), "byte_identical": first == second}
    resolved_config = asdict(config)
    resolved_config["output_dir"] = "<temporary-directory>"
    manifest = {
        "synthetic": True, "verification": "Default generator reproduction, not empirical validation",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "config": resolved_config, "seed": config.seed,
        "generator_sha256": sha256(generator),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "files": entries, "all_byte_identical": all(entry["byte_identical"] for entry in entries.values()),
    }
    (root / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, entry in entries.items():
        print(f"{name}: {entry['row_count']} rows; byte-identical={entry['byte_identical']}")
    return 0 if manifest["all_byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
