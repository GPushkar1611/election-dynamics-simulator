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

## Issue-aware support and issue-specific events

Scalar support remains the default. The opt-in
`synthetic_issue_profile` mode compares each candidate's five positions with a
synthetic five-issue voter preference state weighted by each voter's `issue_*`
priorities. The data contains no measured voter issue stances, so generated
profiles are sensitivity assumptions, not observed beliefs. Their offsets
and final values are saved in `final_state.csv`; when recording history, the
runtime state is also stored as `issue_preferences` in `history.npz`.

Neutral events on a named issue update that issue only in issue-aware mode.
General events still move scalar opinion and therefore shift the profile's
broad center. Candidate-associated events change that candidate's utility.
The rules and limitations are recorded in [ISSUE_PREFERENCE_STATE.md](../ISSUE_PREFERENCE_STATE.md).

Run the issue-aware configuration with:

```powershell
python -m election_sim --config configs/issue_aware_synthetic.json --seed 42 --output results/issue_aware_seed42
```

Compare scalar and issue-aware modes over matched seeds and dispersion values
with `--support-sensitivity --support-seed-count 30`. Add
`--agents campaign_agents/c01_agents.json` to run the same candidate agents in
every arm; agent polls and decisions are saved in
`per_seed_agent_decisions.csv` so their responses can be compared across
support modes. Keep the agent file fixed across arms. All outputs are synthetic
and should be interpreted as sensitivity analysis.

## What “30 days” means

These are 30 simulated days, not 30 real days. Each simulated day updates voter opinions, recalculates candidate support, and records metrics. After day 30, the program calculates turnout and votes. A run finishes in seconds.

## Verification

- All 128 automated tests pass (117 simulator and 11 loader tests).
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

Use a new output folder for each run, or add `--overwrite` to replace a previous run's folder. Run `python -m election_sim --help` for the frozen-control and test commands.

## Full per-day history (for animations)

By default a run saves only 100 sampled voters per day (`sampled_state.csv`). To record **every** voter on **every** day, add `--record-history`:

```powershell
python -m election_sim --config configs/campaign.json --campaign-actions campaign_actions/smear_c01.json --seed 42 --record-history --output results/smear_history_seed42
```

This adds two files to the run folder. A full C01 run is about 1 MB.

- `history.npz`: NumPy arrays. Day 0 is the initial state.

  | Array | Shape | Meaning |
  |---|---|---|
  | `voter_ids` | `[voter]` | Voter IDs in canonical (sorted) order; indexes every voter axis |
  | `candidate_ids` | `[candidate]` | Candidate IDs in sorted order |
  | `action_ids` | `[action]` | Campaign action IDs in the order they are applied |
  | `opinions` | `[day, voter]` | Opinion in [-1, 1] at the end of each day |
  | `support` | `[day, voter, candidate]` | Support probabilities (rows sum to 1) |
  | `smear_belief` | `[day, voter, candidate]` | Belief in a smear about that candidate: 0 unaware, 1 believes, 2 rejects |
  | `campaign_exposed` | `[day, voter, action]` | True if that action reached the voter that day |
  | `event_exposed` | `[day, voter]` | True if any active event reached the voter that day (always False when `lambda_event` is 0) |
  | `issue_preferences` | `[day, voter, issue]` | Present only in `synthetic_issue_profile` mode; the five runtime voter preferences used for candidate issue fit |

- `history_timeline.json`: a list of `{day, end_day, kind, id, label, candidate_id}` entries for campaigns (`campaign_ordinary`, `campaign_misleading`, `campaign_smear`, including agent actions), events (`event`), reminders (`event_reminder`) and withdrawals (`event_candidate_withdrawal`), sorted by day. Use it to label plots and videos.

In Python:

```python
from election_sim import simulate, load_history

result = simulate(config, data, seed=42, campaign_actions=actions, record_history=True)
result.history["opinions"]        # numpy array [31, 3000]

history = load_history("results/smear_history_seed42")   # same dict, read from a saved run
```

Recording only copies values. Results are identical with or without it, and this is tested in `tests/test_history.py`.

## Candidate agents (Step 2)

Candidates can also be agents that read a poll every few days and choose their own campaign (`greedy`, `go_negative` or `rebut`) within a budget. Run:

```powershell
python -m election_sim --config configs/agents.json --agents campaign_agents/c01_agents.json --seed 42 --output results/agents_seed42
```

The command prints every agent decision. See `campaign_agents/README.md`. Step 2 also added the `misleading` message type and optional smear spreading through the social network (`smear_spread_probability`); see `campaign_actions/README.md`. With no agents and spreading off, results are identical to earlier versions.

## Strategic voting (epistemic logic)

With `strategic_voting` > 0, a public `poll_release` makes the poll common knowledge, and voters lower their utility for candidates outside the poll's top two. Candidate agents' private polls never trigger this. See `event_scenarios/STRATEGIC_VOTING.md`.

## What remains

The voter/social-network baseline, ordinary/misleading/smear campaigns, targeted recipients, candidate agents, and optional full history are implemented. Events apply on schedule and scope; candidate-associated events change candidate utility, while neutral issue-specific events update only their issue dimension in issue-aware mode. General neutral events still move scalar opinion. Paired event, factorial event, and issue-support sensitivity runners are available. Issue-aware support uses generated voter issue profiles because the data has priorities but no measured issue stances; see `ISSUE_PREFERENCE_STATE.md`. These remain synthetic mechanisms, not fitted behavioral parameters. Unfinished work includes named semantics and calibration for issue coordinates and shock directions, turnout response, multi-constituency spillover, broader batch experiments, and dashboard/report work. Community labels are inferred from the network; they are not present in voter records. The original plan also calls for model/data-profile documentation and clean-environment reproduction.

The supplied CSV files remain synthetic and are not a validated prediction of a real election.
