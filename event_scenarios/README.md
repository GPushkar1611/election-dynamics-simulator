# Example event schedules

`c01_moderate.csv` is a reviewable 30-day scenario containing 13 rows that
affect constituency 1: 6 constituency-scoped, 3 region-scoped, and 4
state-scoped events. The supplied generated schedule remains unchanged at 278
state-wide rows; this scenario is an alternate schedule, not a regenerated
dataset or a claim about real-world event frequency.

The schedule demonstrates the optional behavior fields:

- a directional healthcare shock (`issue_direction=0.75` shifts only
  healthcare preference in issue-aware mode; scalar mode moves scalar opinion),
- one event conditioned on an earlier event,
- a reminder and finite memory half-life, and
- a local candidate withdrawal.

Run it from the repository root with:

```powershell
python -m election_sim --config configs/baseline.json `
  --events-file event_scenarios/c01_moderate.csv --seed 42 `
  --output results/c01_moderate_events_seed42
```

Add `--overwrite` to run again into the same folder.

For a multi-seed paired comparison, add `--event-seed-count 30` and choose a
new output folder. Add `--event-factorial` to separate withdrawals,
candidate-associated events, and neutral events into matched arms. These
experiments save per-seed candidate and voter-group summaries. For example:

```powershell
python -m election_sim --config configs/baseline.json `
  --events-file event_scenarios/c01_moderate.csv --event-factorial `
  --event-seed-count 30 --seed 0 --output results/c01_event_factorial
```

In issue-aware mode, a neutral event on a specific issue updates only that
preference dimension. `issue_direction` is the preferred directional field;
older schedules may use `opinion_direction` as a fallback. General events
continue to move scalar opinion, and candidate-associated events change
candidate utility directly. In scalar mode, issue-specific neutral events
retain the older scalar-opinion behavior.

All parameters are synthetic scenario choices. In particular, the opinion
axis has no empirically established interpretation as a specific ideology;
the directional sign is supplied explicitly rather than inferred from event
text.
