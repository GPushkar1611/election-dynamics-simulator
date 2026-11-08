# Next Steps: MAS Layer and Final Simulation Notebook

End goal: one final notebook that runs the simulations and renders a **"k-means style" video**. It shows constituencies and their voters, campaigns switching on over time, and the visible effect of campaigns and news on voter opinions.

Real-data benchmarking is out of scope here. This document covers the multi-agent system work only.

---

## 1. Current status (as of 2026-11-08)

The simulator now includes Steps 1–5: full history, candidate agents/campaigns, news-driven turnout, regional spillover, and a matched-seed experiment workflow. On this revision, 140 simulator tests and 11 loader tests pass. The checked-in datasets are synthetic; results are scenario demonstrations, not forecasts.

| Area | Status |
|---|---|
| Loader and validation (single constituency, internal edges only) | Done |
| Directed bounded-confidence DeGroot, synchronous, seeded | Done |
| Campaigns: `ordinary`, `misleading`, and `smear` | Done |
| Targeting: uniform, group, high susceptibility, high influence, community | Done |
| Events: scope, decay, reminders, dependencies, withdrawal, paired shock test | Done |
| Turnout with optional campaign/news exposure coefficients and voting | Done |

## 2. Gaps (MAS point of view)

1. ~~Candidates are not agents; campaign messages lack misleading labels or smear spread.~~ Fixed in Step 2.
2. ~~Turnout ignores campaigns and events.~~ Step 3 connects cumulative campaign/news exposure to turnout; both coefficients remain zero by default.
3. ~~No multi-seed runner or paired campaign comparison.~~ Step 5 provides `run_batch`, `compare_campaign`, median and 5th–95th percentile summaries.
4. ~~Single constituency only.~~ Step 4 adds regional social influence with controlled spillover; external region boundaries remain excluded.
5. **No visualization or animation code.** `matplotlib` is not in `requirements.txt`.
6. **No final animation notebook.** Step 5 selected C21 with a 4.33-point top-two support gap; the video and notebook still need to be built around this scenario.
7. **`docs/model.md` and `docs/decisions.md` are missing.**

---

## 3. Steps

### Step 1: Record full history for animation (A, E) - DONE
Implemented in `election_sim/simulation.py` and tested in `tests/test_history.py`. Usage and file format are documented in `election_sim/BASELINE_SIMULATION.md`. The original plan for this step follows.

- Add `record_history: bool = False` to `simulate()`. When it is true, return:
  - `opinions`: float32 `[days+1, N]`
  - `support`: float32 `[days+1, N, C]`
  - `campaign_exposed`: bool `[days+1, N, A]` (one exposure column per campaign action)
  - `event_exposed`: bool `[days+1, N]`
  - `timeline`: a list of `{day, kind, label, candidate_id}` covering campaign starts and ends, events, smears and fact-checks
- Save it as `history.npz`, not CSV.
- Add a test that history at day 30 matches `final_state`.

### Step 2: Make candidates real agents (B, D) - DONE
Implemented in `election_sim/agents/` (`poll.py`, `candidate.py`), `election_sim/campaign/` (misleading type, smear spreading) and `election_sim/simulation.py`. Tests: `tests/test_agents.py`, plus new cases in `tests/test_campaign.py`. Usage: `campaign_agents/README.md`. Differences from the plan below:
- "scripted" is not a separate strategy: scripted campaigns are still the JSON files in `campaign_actions/`, and they can run alongside agents.
- Smears are one claim per target candidate: all smears about the same candidate share one belief state.

The Step 5 experiment matrix now compares scripted campaigns with `greedy`, `go_negative`, and `rebut` agents under a common per-candidate contact budget and matched seeds.

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

### Step 3: News agent and turnout (C) - DONE
Implemented in `election_sim/news.py`, `simulation.py` and `voting.py`.
Usage, assumptions and runnable example: `event_scenarios/STEP_3.md`.
Tests: `tests/test_news.py`. Strategic voting remains future work.
- Wrap `events.py` in a `NewsAgent` that can read the CSV schedule and also generate seeded random events, for example a Poisson number per day.
- New event types:
  - `fact_check`: moves exposed believers of a smear to "rejects". Credibility sets the acceptance probability.
  - `poll_release`: publishes the current poll to everyone (used by strategic voting; see §5).
- Connect turnout mobilization, keeping the gammas at 0 by default:
  `p_i = sigmoid(logit(propensity_i) + gamma_mob * campaign_exposure_i + gamma_event * event_exposure_i)`

### Step 4: Multiple constituencies (A, E) - DONE
Implemented in `election_sim/region.py` using the shared daily simulation engine.
Usage and assumptions: `STEP_4.md`. CLI: `python -m election_sim.region_cli`.
Tests: `tests/test_region.py`; paired single-seed spillover sweep available.
Cross-boundary coupling applies to social opinions; smear claims remain local.
- Add `simulate_region(config, constituency_ids, seed, spillover=0.0)`.
- Simulate all of the region's voters as one array. Keep cross-constituency edges, but multiply their weight by `spillover`. With `spillover=0` the constituencies are independent, which acts as the control.
- Region 1 has 7 constituencies, about 21,000 voters, which is still fast with `np.bincount`.
- Run agents per constituency. Map party-level campaigns to each constituency's candidate through `party_id`.
- Results come per constituency, plus a seats table.

### Step 5: Experiments (D, E) - DONE
Implemented in `election_sim/experiments.py` (`run_batch`, `compare_campaign`, `summarize_batch`, `build_step5_scenarios`) and `election_sim/step5_cli.py`. Tests: `tests/test_batch_experiments.py`. The runner accepts named scenarios, uses common seed values across arms, and emits one row per (scenario, seed, constituency, candidate). `compare_campaign` changes only the scripted actions and/or candidate agents while keeping config, data, and seed fixed.

The standard matrix includes frozen and social controls; ordinary, misleading, and smear campaigns; social-plus-news and combined runs; scripted versus `greedy`, `go_negative`, and `rebut` agents; and equal-reach uniform, student-group, high-susceptibility, high-influence, and community targeting. Campaign strategy arms share a per-candidate contact-budget cap; targeted arms request the same reach. The core run diagnostics report audience shortfalls and agent budget use, but the current batch CSVs do not retain those per-seed diagnostics, so verify actual contact use separately before interpreting a budget comparison. The workflow first scans all constituency baselines with campaigns disabled and the configured event schedule retained, ranks by top-two mean-support gap, selects the closest one, and runs the matrix over 30 seeds by default. It also runs the regional spillover values 0, 0.05, 0.1, and 0.2 over the same seed list. Reports contain medians and empirical 5th–95th percentile ranges; those ranges describe the requested synthetic runs only.

Completed run: `results/step5_seed0_29/`. The baseline scan used seed 0 across all 21 constituencies and selected C21, where the top-two mean-support gap is 0.0433 (4.33 percentage points). The selected-race matrix has 16 arms and seeds 0–29; the regional sweep covers all 7 constituencies in C21's region at all four spillover values. The output bundle includes per-seed outcomes, summaries, paired effects, and the selection/seed manifest. In this particular scenario, CAND_0061 won all 30 `social_only` runs; this is a property of the synthetic setup, not an election forecast. The closest-race criterion is baseline support, while vote outcomes still vary by seed.

Run from the repository root (use a new output directory):

```powershell
python -m election_sim.step5_cli --seed 0 --seed-count 30 --output results/step5_seed0_29
```

Outputs include `baseline_scan.csv`, `per_seed.csv`, `summary.csv`, `paired_effects.csv`, `spillover_per_seed.csv`, `spillover_summary.csv`, and `manifest.json`. The manifest records the selected constituency and whether its baseline support gap is within 10 percentage points. If no baseline is that close, the workflow selects the closest available and reports that fact; it does not silently tune candidate popularity.

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
- [x] `NewsAgent` with fact-check and poll release; mobilization connected
- [x] `simulate_region` with spillover sweep
- [x] `run_batch`, `compare_campaign`, the Step 5 scenario matrix and 30-seed results
- [x] Baseline scan across all constituencies and closest-race selection
- [ ] `viz.py` renders an MP4 of several constituencies with campaign and event overlays
- [ ] `notebooks/final_simulation.ipynb` runs top to bottom on a clean install
- [ ] `docs/model.md`, `docs/decisions.md`, report
