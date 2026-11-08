# Step 3: News agent and turnout

All data and behavioral parameters are synthetic scenario assumptions.

From the repository root:

```powershell
python -m election_sim --config configs/news.json --campaign-actions campaign_actions/smear_c01.json --events-file event_scenarios/c01_news.csv --seed 42 --record-history --output results/news_seed42
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s constituency_loader -p "test_*.py"
```

This example smears CAND_0003 on days 10–15, fact-checks it on day 18,
applies a neutral shock on day 20, and publishes a poll on day 25. The
config also generates Poisson(0.1) neutral events per day. Set
`random_news_rate` to 0 to use only the CSV schedule. The nonzero gammas
in this demonstration are illustrative; both default to zero.

## Python API

```python
from election_sim import NewsAgent, simulate
news = NewsAgent("event_scenarios/c01_news.csv", random_events_per_day=0.1)
result = simulate(config, data, seed=42, news_agent=news, record_history=True)
public_poll = result.diagnostics["public_polls"][-1]
```

Without an explicit NewsAgent, the simulation wraps data["events"] and uses
config.random_news_rate. With an explicit agent, its schedule and random rate
are authoritative. A schedule can also be a DataFrame. Each run compiles fresh
state, so the same agent can be reused. Random schedules and event-specific
acceptance/poll draws use the event seed stream, leaving private candidate
polling, turnout and voting streams unchanged. Generated events use reserved
`random-news-{day}-{index}` IDs; collisions with scheduled IDs are rejected.

## Event semantics

Existing shock types, scope, groups, dependencies, occurrence probabilities,
durations, decay and reminders retain their existing behavior.

- `fact_check`: candidate_id names the local candidate targeted by the false
  smear claim. Use candidate_effect=0. On each active day, exposed believers
  independently accept with probability credibility and become REJECTS.
  This announcement does not also apply an opinion/utility shock. Acceptance
  is independent of event sensitivity and intensity. Rumour spreading cannot
  reinfect rejecters; a later direct scripted smear can convince them again
  under the existing direct-smear rule.
- `poll_release`: use target_group=General, no candidate_id, and blank
  candidate_effect. After that day's support update, a fresh noisy sample
  using poll_sample_size/poll_noise is published to everyone. The entire
  electorate receives the same announcement; credibility does not restrict
  delivery. The poll is recorded with day, event ID, shares and public status.
  Strategic voting is future work: publication itself does not change support.
- Announcement types do not decay, trigger reminders, or withdraw candidates.
  Duration greater than one repeats the announcement each active day.
- lambda_event=0 disables all news effects, announcements and news exposure.

Daily order: private candidate polling/decisions, social update, campaigns,
smear spread, news (including corrections), support recalculation, public
poll publication, history. Timeline entries distinguish fact_check and
poll_release; belief history shows corrections on the day they occur.

## Turnout

`p = sigmoid(logit(propensity) + gamma_mob * campaign_exposure + gamma_event * event_exposure)`

Exposure means cumulative voter-days: each voter counts at most once per
channel per day, even with overlapping actions/events. Campaign exposure
counts direct messages (ordinary, misleading, smear), not rumour relays.
News exposure includes active shocks, announcements and applied reminders;
memory decay alone does not count as fresh exposure. News exposure is delivery,
not acceptance or nonzero sensitivity. Coefficients can be positive or negative.

For a nonzero shift, baseline propensity is clamped to [1e-6, 1-1e-6] before
logit. Zero shifts preserve the exact original propensity, including 0 and 1,
so default turnout and paired random draws remain compatible.

Inspect final_state.csv for campaign_exposure, event_exposure and
turnout_probability. manifest.json includes config, seed streams, public_polls,
per-event applications/acceptances, and the random event rate. Full history
includes actual news delivery, including reminders.
