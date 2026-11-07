# Next Steps: MAS Layer and Final Simulation Notebook

End goal: one final notebook that runs the simulations and renders a **"k-means style" video**. It shows constituencies and their voters, campaigns switching on over time, and the visible effect of campaigns and news on voter opinions.

Real-data benchmarking is out of scope here. This document covers the multi-agent system work only.

---

## 1. Current status (as of 2026-11-07)

All 98 simulator tests and 10 loader tests pass. One C01 run (3,000 voters, 30 days) takes about 2 s.

| Area | Status |
|---|---|
| Loader and validation (single constituency, internal edges only) | Done |
| Directed bounded-confidence DeGroot, synchronous, seeded | Done |
| Campaigns: `ordinary` and `smear` | Done |
| Targeting: uniform, group, high susceptibility, high influence, community | Done |
| Events: scope, decay, reminders, dependencies, withdrawal, paired shock test | Done |
| Turnout (propensity only) and voting | Done |

## 2. Gaps (MAS point of view)

1. ~~Candidates are not agents.~~ Fixed in Step 2: candidate agents poll, decide and act within a budget.
2. ~~No `misleading` message type; smears do not spread.~~ Fixed in Step 2.
3. **Turnout ignores campaigns and events.** `gamma_mob` and `gamma_event` from plan §3.6 are not connected.
4. **No multi-seed runner, and no paired campaign comparison.** Only events have `compare_event_shock`.
5. **Single constituency only.** 5,533 cross-constituency edges are dropped, and there is no spillover.
6. ~~Only 100 sampled voters per day are saved.~~ Fixed: use `record_history=True` / `--record-history` (Step 1).
7. **No visualization or animation code.** `matplotlib` is not in `requirements.txt`.
8. **The C01 baseline is a landslide.** CAND_0001 gets about 67%, so campaigns won't visibly change the result. The demo needs a closer race.
9. **`docs/model.md` and `docs/decisions.md` are missing.**

---

## 3. Steps

### Step 1: Record full history for animation (A, E) - DONE
Implemented in `election_sim/simulation.py` and tested in `tests/test_history.py`. Usage and file format are documented in `election_sim/BASELINE_SIMULATION.md`. The original plan for this step follows.

- Add `record_history: bool = False` to `simulate()`. When it is true, return:
  - `opinions`: float32 `[days+1, N]`
  - `support`: float32 `[days+1, N, C]`
  - `campaign_exposed`: int `[days+1, N]` (index of the action reaching the voter, -1 for none)
  - `event_exposed`: bool `[days+1, N]`
  - `timeline`: a list of `{day, kind, label, candidate_id}` covering campaign starts and ends, events, smears and fact-checks
- Save it as `history.npz`, not CSV.
- Add a test that history at day 30 matches `final_state`.

### Step 2: Make candidates real agents (B, D) - DONE
Implemented in `election_sim/agents/` (`poll.py`, `candidate.py`), `election_sim/campaign/` (misleading type, smear spreading) and `election_sim/simulation.py`. Tests: `tests/test_agents.py`, plus new cases in `tests/test_campaign.py`. Usage: `campaign_agents/README.md`. Differences from the plan below:
- "scripted" is not a separate strategy: scripted campaigns are still the JSON files in `campaign_actions/`, and they can run alongside agents.
- Smears are one claim per target candidate: all smears about the same candidate share one belief state.

Still open for D: a fixed-budget comparison of strategies (scripted vs greedy vs go_negative) over many seeds belongs to Step 5 (`run_batch`, `compare_campaign`).

- New module `election_sim/agents/candidate.py` with this class:
  ```python
  class CandidateAgent:
      def __init__(self, candidate_id, strategy, budget, rng): ...
      def observe(self, poll: dict) -> None: ...          # noisy poll only, no direct state access
      def decide(self, day: int) -> list[CampaignAction]: ...
  ```
- **Polls:** sample about 300 voters each day (or every k days), add noise, and pass the result to `observe`. Agents never see true opinions.
- **Strategies:**
  - `scripted`: the current JSON. Keep it for control runs.
  - `greedy`: target the group where the candidate's poll support gap is largest.
  - `go_negative`: if behind the leader by more than X, smear the leader.
  - `rebut`: answer a smear aimed at itself with an ordinary or fact-check message.
- **Budget:** a fixed total number of voter contacts per candidate across the campaign, so comparisons are fair.
- **Simulation loop:** each day, call the agents' `decide()` before the campaign step. Include the day in `action_id` so recipient selection stays reproducible.
- **`misleading` type:** same update as `ordinary`, but the message position is distorted toward the audience, e.g. `q = platform + distortion * (audience_mean - platform)`. Keep the truth label separate from the mechanics.
- **Smear spread:** give each voter a state `unaware / believes / rejects`. Belief passes along edges with probability `p * edge_weight * credibility`. The utility penalty applies only while the voter believes the claim.

### Step 3: News agent and turnout (C)
- Wrap `events.py` in a `NewsAgent` that can read the CSV schedule and also generate seeded random events, for example a Poisson number per day.
- New event types:
  - `fact_check`: moves exposed believers of a smear to "rejects". Credibility sets the acceptance probability.
  - `poll_release`: publishes the current poll to everyone (used by strategic voting; see §5).
- Connect turnout mobilization, keeping the gammas at 0 by default:
  `p_i = sigmoid(logit(propensity_i) + gamma_mob * campaign_exposure_i + gamma_event * event_exposure_i)`

### Step 4: Multiple constituencies (A, E)
- Add `simulate_region(config, constituency_ids, seed, spillover=0.0)`.
- Simulate all of the region's voters as one array. Keep cross-constituency edges, but multiply their weight by `spillover`. With `spillover=0` the constituencies are independent, which acts as the control.
- Region 1 has 7 constituencies, about 21,000 voters, which is still fast with `np.bincount`.
- Run agents per constituency. Map party-level campaigns to each constituency's candidate through `party_id`.
- Results come per constituency, plus a seats table.

### Step 5: Experiments (D, E)
- Add `run_batch(scenarios, seeds) -> tidy DataFrame` with one row per (scenario, seed, constituency, candidate).
- Add `compare_campaign()`, modelled on `compare_event_shock`: same seeds, only the campaign differs.
- Scenario matrix: frozen, social only, with campaigns (ordinary, misleading, smear), with news, all together, strategic agents compared with scripted ones, and equal-budget targeting.
- Use 3–5 seeds while debugging and 30 for the final results. Report the median and the 5th–95th percentile range.
- Spillover sweep: 0, 0.05, 0.1, 0.2.
- **Pick or tune a close race for the demo.** Scan all 21 constituencies' baselines and choose one where the top two candidates are within about 10 points.

### Step 6: `election_sim/viz.py` and the final video
**Layout (k-means style):**
- Each constituency is a tile, laid out on a grid.
- A tile's candidates are fixed corner points: an equilateral triangle for 3 candidates, a regular polygon for k. Draw them as large markers in each candidate's colour.
- Voter position on day t: `pos_i(t) = sum_c support_ic(t) * vertex_c + fixed_jitter_i`. Undecided voters sit near the centre, and committed voters slide toward a corner, so clusters form like k-means iterations.
- Colour shows `argmax support` (the cluster). Opacity shows `max support` (how committed the voter is).
- Interpolate about 8 frames per simulated day so the motion is smooth.

**Overlays:**
- Day counter and scenario title.
- A timeline bar at the bottom showing campaign spans (in candidate colours) and event ticks.
- Voters a campaign reaches get a ring in the candidate's colour while it is active.
- Event banner text (e.g. "Day 20: Healthcare shock") and a brief flash on exposed voters.
- Smear believers are marked, e.g. with an x marker or dark outline. Fact-checks visibly clear them.
- A side panel with a support-over-time line chart that grows as the video plays. At the end, show vote bars with turnout and the winner.
- **Side-by-side:** the control run (no campaigns) next to the campaign run, using the same seed, so the effect is visible.

**Rendering:**
- matplotlib `FuncAnimation`, updating one scatter per tile with `set_offsets` and `set_facecolors`.
- Save with `anim.save("demo.mp4", writer="ffmpeg", fps=20, dpi=150)`. Fall back to `pillow` for a GIF.
- Display in the notebook with `IPython.display.Video("demo.mp4", embed=True)`.
- Add `matplotlib` and `ipykernel` to `requirements.txt`. `ffmpeg` must be on PATH.
- Optional: a Plotly animation for an interactive version.

**Final notebook (`notebooks/final_simulation.ipynb`):**
1. Setup, with a visible warning that the data is synthetic.
2. Load the region and choose the demo constituencies.
3. Define the agents and the scenario timeline:
   - Days 1–9: social influence only (campaigns off).
   - Day 10: candidate agents switch on.
   - Day 15: the trailing candidate goes negative with a smear.
   - Day 18: fact-check news.
   - Day 20: a neutral news shock.
   - Day 25: poll release (public announcement).
   - Day 30: election.
4. Run the paired simulations (control and treatment) and the multi-seed summary.
5. Static figures: support over time, opinion histograms, polarization, targeting comparison.
6. The video.
7. Results table: vote shares, turnout, winner, margin, and effect sizes with seed intervals.
8. Epistemic scenario (§5): share of voters believing the false claim over time, and poll release with and without strategic voting.

### Step 7: Docs and report (all; E integrates)
- `docs/model.md`: all update rules, the agent loop, and the order of steps within a day.
- `docs/decisions.md`: parameter defaults and assumptions.
- Report sections: model, agents, experiments, video stills, limitations (synthetic data).

---

## 4. Where epistemic logic fits

Not needed for the core model: DeGroot and bounded-confidence dynamics model *degrees of opinion*, not *knowledge*. It does fit naturally in three places:

1. **Misinformation (belief versus knowledge).**
   - Let φ = "CAND_0003 is corrupt", which is false in the actual world.
   - A smear makes recipients *believe* φ (B_i φ). They cannot *know* it (K_i φ), because knowledge requires truth.
   - Voter states: unaware, B_i φ, or B_i ¬φ.
   - The claim spreads along edges, which is gossip.
   - A fact-check is an announcement of ¬φ to one group, a group announcement in dynamic epistemic logic. Credibility makes it a soft update rather than a hard one.
   - Metric: the share of voters believing a falsehood over time.
2. **Partial observability of candidates.**
   - Candidates only know poll results. An internal poll is private knowledge of one candidate.
   - A published poll is a public announcement, so its result becomes common knowledge.
   - Strategy depends on what each agent knows, and what it knows others know.
3. **Strategic voting through common knowledge.**
   - If a published poll makes it common knowledge that C2 cannot win, C2 supporters defect to their second choice (Duverger's law). Without a published poll, nobody defects.
   - This depends on higher-order beliefs: "I know that everyone knows C2 is losing."
   - The same mechanism supports stretch goal 2: early-phase results, once publicly announced, become common knowledge that shapes later phases.

**Implementation advice:**
- Do not build full Kripke models over 3,000 agents; the number of possible worlds explodes.
- In the simulator, use belief flags plus three announcement types: public, group and private.
- In the report, formalize a small 3-agent example with Kripke models, showing how a public announcement versus a private one updates knowledge.

---

## 5. Definition of done
- [x] `simulate(..., record_history=True)` returns the full arrays, with a test
- [x] At least 3 candidate strategies, polls with noise, budgets enforced
- [x] `misleading` type and smear spread, with tests
- [ ] `NewsAgent` with fact-check and poll release; mobilization connected
- [ ] `simulate_region` with spillover sweep
- [ ] `run_batch` and `compare_campaign`, 30-seed results
- [ ] `viz.py` renders an MP4 of several constituencies with campaign and event overlays
- [ ] `notebooks/final_simulation.ipynb` runs top to bottom on a clean install
- [ ] `docs/model.md`, `docs/decisions.md`, report
