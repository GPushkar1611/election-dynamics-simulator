# Constituency data loader

This folder loads the supplied synthetic CSV data for one constituency. It prepares a clean slice for a simulation; it does not run a simulation.

## Files

- `loader.py` reads the four CSV files, selects voters and candidates by constituency ID, keeps only social ties whose source and target are both local, selects local/region/state events, and reports excluded boundary ties.
- `validation.py` checks IDs, references, numeric ranges, edge rules, and event geography. It raises `DataValidationError` with the problems it finds.
- `__init__.py` lets Python import this folder as a package and exposes `load_constituency` and the validation helpers.
- `test_loader.py` has small checks for loading C01 and handling an unknown constituency.
- `example.ipynb` walks through loading and inspecting the returned data.

## Use it

Install the project's data dependencies if needed:

```bash
python -m pip install pandas
```

From the project root, Python can import a package only if its directory name is a valid Python identifier. Since this folder currently contains a space, the simplest use is to add the folder to `sys.path` and import `loader`:

```python
import sys
from pathlib import Path

project = Path.cwd()  # run from the project root
sys.path.insert(0, str(project / "constituency loader"))
from loader import load_constituency

data = load_constituency(project / "data" / "synthetic_election_dataset", constituency_id=1)
voters = data["voters"]
candidates = data["candidates"]
edges = data["social_network"]
events = data["events"]
print(data["diagnostics"])
```

The result is a dictionary with four pandas DataFrames and a `diagnostics` dictionary. Constituency IDs are the numeric IDs in the CSV files; `1` is C01. Set `validate=False` only when you intentionally want to skip input checks.

## Run the checks

From the project root:

```bash
python -m unittest discover -s "constituency loader" -p "test_*.py" -v
```

The checks use the supplied dataset. The notebook has the same basic example in an interactive format.
