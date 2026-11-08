# Election Dynamics Simulator

A reproducible, agent-based simulation of voter opinion, campaigns, news, and elections across synthetic constituencies. The project is for exploring model behavior; it is **not a forecast of a real election**.

## Quick start

Use Python 3.12.10 for the recorded dataset-reproduction environment. Install dependencies and run a baseline simulation:

```powershell
python -m pip install -r requirements.txt
python -m election_sim --config configs/baseline.json --seed 42 --output results/baseline
```

Run the final notebook from the repository root:

```powershell
jupyter lab final_simulation.ipynb
```

The notebook compares a no-campaign control with an intentionally strong campaign scenario, plots the changes, and creates an animation. Its campaign parameters are demonstration settings chosen to make effects visible; they are not calibrated estimates.

## What the model does

- Voters update a scalar opinion through directed, bounded-confidence social influence.
- Candidate platforms and popularity determine voter support.
- Scripted campaigns and polling candidate agents can change opinions or candidate utility.
- Scheduled events can change opinions, support, turnout, or candidate status.
- Optional public polls trigger a simplified strategic-voting response.
- Runs can cover one constituency or a region with social spillover.

All supplied voter, candidate, network, and event data are synthetic assumptions. See [the model guide](docs/model.md) for update rules and limitations.

## Repository map

| Path | Purpose |
|---|---|
| `election_sim/` | Simulation, agents, campaigns, events, voting, and experiment runners |
| `constituency_loader/` | CSV loading and validation |
| `data/` | Synthetic dataset, generator, and data notes |
| `configs/` | Simulation parameter sets |
| `campaign_actions/`, `campaign_agents/`, `event_scenarios/` | Example inputs |
| `tests/` | Simulator regression tests |
| `final_simulation.ipynb` | Final visual walkthrough |
| `docs/` | Model, workflow, and project notes |
| `references/` | Research and course references |

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s constituency_loader -p "test_*.py" -v
python data/verify_dataset.py
```

## More information

- [Getting started and commands](docs/guide.md)
- [Model rules and limitations](docs/model.md)
- [Final animation notebook](final_simulation.ipynb)
- [Data and synthetic-data caveats](data/README.md)
- [Candidate campaigns](campaign_actions/README.md) and [candidate agents](campaign_agents/README.md)
