# My Contribution: Voter and Social-Network Baseline

This contribution implements the baseline simulation described in the original `PROJECT_IMPLEMENTATION_PLAN.md`.

## What was fixed

The constituency loader was including local events from other constituencies in the same region. This could mix unrelated events into the selected constituency.

The loader now includes only:

- Local events belonging to the selected constituency.
- Regional events belonging to its region.
- State-wide events.

For C01, this gives 26 events: 12 local, 6 regional, and 8 state-wide. These events are loaded but do not yet affect the baseline simulation.

Input checks were also strengthened to reject invalid values, categories, event dates, and inconsistent candidate/event geography.

## What was implemented

- Voter opinions and candidate support probabilities based on candidate popularity and policy positions.
- Directed social influence: a source voter influences a target voter.
- Bounded confidence: voters accept influence only within their confidence threshold.
- Synchronous updates: everyone updates from the same previous-day opinions.
- A reproducible 30-day simulation using separate random streams.
- Basic turnout, vote sampling, candidate vote totals/shares, winners, ties, and vote accounting.
- Frozen-opinion and social-only configurations.
- Daily opinion and support metrics, opinion distributions, and a small daily sample of voter states.
- Saved results containing configuration, seeds, data/code hashes, and metadata.

## What “30 days” means

These are 30 simulated days, not 30 real days. Each simulated day updates voter opinions, recalculates candidate support, and records metrics. After day 30, the program calculates turnout and votes. A run finishes in seconds.

## Verification

- All 58 automated tests pass (48 simulator and 10 loader tests).
- C01 was checked with seeds 42, 43, and 44.
- Shuffling input rows produces exactly the same results for the same seed.
- All 3,000 first-day C01 opinions match an independent calculation.
- The frozen control leaves initial opinions unchanged.
- Votes and abstentions correctly add up to the voter population.
- Regenerating the supplied dataset with seed 42 produces four byte-identical CSV files.

## Main files

The project is in the repository root.

- `constituency_loader/`: loader correction, stronger validation, and regression tests.
- `election_sim/`: configuration, graph, support, social updates, voting, campaign targeting, simulation, and command-line runner.
- `campaign_actions/`: ordinary/smear action examples and targeted-audience examples.
- `configs/baseline.json` and `configs/frozen.json`: the two baseline scenarios.
- `tests/test_baseline.py`: baseline verification tests.
- `data/verify_dataset.py` and `data/manifest.json`: dataset reproduction check and its recorded results.
- `requirements.txt`: pinned dependencies.
- `.gitignore`: excludes generated results, caches, and the virtual environment.

## How to run

From the project folder, with the dependencies installed:

```powershell
python -m election_sim --config configs/baseline.json --seed 42 --output results/my_baseline_run
```

Use a new output folder for each run. Run `python -m election_sim --help` for the frozen-control and test commands.

## What remains

The voter/social-network baseline, ordinary/smear campaign mechanics, and uniform, group, high-susceptibility, high-influence, and seeded graph-community recipient policies are implemented. Scheduled event rows are applied on their active days: group, issue importance, and event sensitivity filter exposure; candidate-associated events adjust that candidate's utility, while candidate-free events support balanced or explicitly directional opinion shocks. Effects are apportioned over duration and can decay, receive reminders, depend on an earlier event, or trigger candidate withdrawal. A 13-event C01 example schedule demonstrates these options without changing the 278-row generated dataset. The paired shock/no-shock helper holds random streams constant. These are synthetic mechanisms, not fitted behavioral parameters. Multi-seed experiment workflows, mappings from shock direction to named political responses, issue-aware candidate support, uncertainty analysis, and the dashboard/report remain unfinished. Community labels are inferred from the network; they are not present in voter records. The original plan also calls for model/data-profile documentation and clean-environment reproduction.

The supplied CSV files remain synthetic and are not a validated prediction of a real election.
