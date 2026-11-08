# Live Election Observatory

A local, browser-based, day-by-day view of the existing synthetic election simulator. It uses the repository's data validator, simulation configuration, candidate agents, campaign action models, social update, scheduled events, and voting engine. It does not replace the normal CLI or change its output files.

## Run it

From the repository root, with the dependencies in `requirements.txt` installed:

```powershell
python -m live_simulation
```

Open `http://127.0.0.1:8765`. The server binds to localhost by default. Choose a constituency to focus on, a scenario, and a seed, then start a run. The state view advances all 21 constituency engines together against the existing statewide social graph. The main canvas draws every voter in every constituency; the constituency view zooms into all voters in one seat. Territory outlines are invented schematic geometry because the synthetic dataset has no real geographic boundaries. Voter positions inside those territories are stable display coordinates, not real locations.

## Scenarios

- **Full campaign** runs the supplied event schedule with local candidate agents using the repository's greedy, go-negative, and rebut strategies.
- **Scripted campaign** adds a seeded ordinary campaign in every constituency using the existing `CampaignAction` and campaign-effect implementation.
- **Social control** runs social dynamics without scheduled news or campaigns.

The run streams daily snapshots through server-sent events. Each constituency payload includes every voter's opinion, candidate support, and source deltas in a compressed binary block. The browser renders these agents on Canvas and retains each day for replay, zoom, pan, and voter inspection. Opinion deltas are separated into social influence, campaign messages, and neutral event signals. Candidate-linked events, smear beliefs, polls, and other non-opinion changes appear in the activity or support views. Preference-switch totals and previous-to-new candidate transitions are computed from each day's actual support arrays.

## Boundaries

All data and results are synthetic. The dashboard is an exploratory visualization, not an election forecast. The observer hook in `election_sim/simulation.py` is optional: standard simulation calls do not collect per-source arrays and retain their existing save formats. The live interface has no third-party web dependencies; it uses Python's standard HTTP server, the existing NumPy/pandas stack, Canvas, and server-sent events.
