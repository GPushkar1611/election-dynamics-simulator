# Constituency data loader

This folder loads the supplied synthetic CSV data for one constituency. It prepares a clean slice for a simulation; it does not run a simulation.

## Files

- `loader.py` reads the four CSV files, selects voters and candidates by constituency ID, keeps only social ties whose source and target are both local, selects local/region/state events, and reports excluded boundary ties.
- `validation.py` checks IDs, references, numeric ranges, edge rules, and event geography. It raises `DataValidationError` with the problems it finds.
- `__init__.py` lets Python import this folder as a package and exposes `load_constituency` and the validation helpers.
- `test_loader.py` has small checks for loading C01 and handling an unknown constituency.
- `test_validation.py` checks invalid ranges/categories, event calendars, candidate geography, and inconsistent event references.
- `example.ipynb` walks through loading and inspecting the returned data.

## Use it

Install the project's data dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

From the project root, import the package directly:

```python
from pathlib import Path
from constituency_loader import load_constituency

project = Path.cwd()  # run from the project root
data = load_constituency(project / "data" / "synthetic_election_dataset", constituency_id=1)
voters = data["voters"]
candidates = data["candidates"]
edges = data["social_network"]
events = data["events"]
print(data["diagnostics"])
```

The result is a dictionary with four pandas DataFrames and a `diagnostics` dictionary. Constituency IDs are the numeric IDs in the CSV files; `1` is C01. Set `validate=False` only when you intentionally want to skip input checks. The supplied dataset's event calendar is 30 days; pass `election_duration_days` explicitly when loading a separately generated dataset with a different calendar. Events may continue beyond the last day; the eventual scheduler must truncate their effects at election close.

## Run the checks

From the project root:

```bash
python -m unittest discover -s constituency_loader -p "test_*.py" -v
```

The checks use the supplied dataset and a small synthetic fixture that verifies local, regional, and state event scopes across multiple constituencies. For C01, the supplied data returns 26 events: 12 local, 6 regional, and 8 state events. Local events from another constituency are excluded even when it shares C01's region.
