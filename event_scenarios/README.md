# Example event schedules

`c01_moderate.csv` is a reviewable 30-day scenario containing 13 rows that
affect constituency 1: 6 constituency-scoped, 3 region-scoped, and 4
state-scoped events. The supplied generated schedule remains unchanged at 278
state-wide rows; this scenario is an alternate schedule, not a regenerated
dataset or a claim about real-world event frequency.

The schedule demonstrates the optional behavior fields:

- a directional state shock (`opinion_direction=0.75` shifts exposed voters
  toward the positive end of the model's opinion scale),
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

All parameters are synthetic scenario choices. In particular, the opinion
axis has no empirically established interpretation as a specific ideology;
the directional sign is supplied explicitly rather than inferred from event
text.
