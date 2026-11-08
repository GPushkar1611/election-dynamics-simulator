# Running the project

Run commands from the repository root.

## Single constituency

```powershell
python -m election_sim --config configs/baseline.json --seed 42 --output results/baseline
```

Use `--campaign-actions campaign_actions/ordinary_c01.json` for a scripted campaign, or `--agents campaign_agents/c01_agents.json` for poll-driven candidate agents. Add `--record-history` to save every voter's daily opinion and support for visualization.

## Region

```powershell
python -m election_sim.region_cli --config configs/baseline.json --constituencies 1 2 3 4 5 6 7 --spillover 0.1 --seed 42 --output results/region
```

See [regional notes](regional-guide.md) for party-level actions and regional outputs.

## Experiments

```powershell
python -m election_sim.step5_cli --seed 0 --seed-count 30 --output results/step5
```

Generated results are ignored by Git. Choose a new output directory for each run. The simulator refuses to overwrite existing output folders except where its CLI explicitly supports `--overwrite`.

## Tests and data verification

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s constituency_loader -p "test_*.py" -v
python data/verify_dataset.py
```

## Important interpretation note

The supplied dataset is synthetic. Run outputs and seed intervals describe the specified model scenarios; they are not real-election predictions or estimates of real-world uncertainty.
