# Election Dynamics Simulator - Complete Implementation Plan

## 1. Goal and scope

Implement the updated proposal's virtual election: voters start with synthetic opinions/attributes, influence one another through a directed social graph, respond to candidate campaigns including explicitly labeled misleading and smear signals, and experience neutral but disruptive news/external events. A turnout rule converts final preferences into votes. The reproducible simulator now supports both single constituencies and selected multi-constituency regions with controlled social spillover. A multi-phase national feedback layer remains stretch work.

**Data status:** The repo has a synthetic data generator and fictional state, not real election results. Supplied data: 3 regions, 21 constituencies, 3,000 voters each (63,000 total), 3 candidates each (63), 506,629 directed edges, and 278 events over 30 days. All values are generated assumptions, not facts about real voters. Use them for mechanics and scenario demonstrations. If the final report compares to a real election, obtain official results separately. Without independent pre-election opinion data, call results a synthetic scenario or aggregate reconstruction, not a validated forecast.

Core outputs: single-constituency and regional simulation; DeGroot/bounded-confidence dynamics; candidate agents with ordinary, misleading and smear campaigns; news/event agents; shared targeting policies; exposure-sensitive turnout; matched-seed experiments; dashboard/figures; report and demo.

Regional social spillover is implemented in Step 4. Stretch only after core: a synthetic multi-phase national layer where early results influence later campaigns.

## 2. Begin with supplied data

### 2.1 Freeze and profile
Generator: data/dataset_generator.py.

Tables: data/synthetic_election_dataset/{voters,social_network,candidates,political_events}.

Defaults: seed 42, 3 regions x 7 constituencies, 3,000 voters and 3 candidates per constituency, 30 days, 12 local events per constituency, 6 regional events per region, 8 state events, mean 8 incoming network ties and target community size 150.

1. Create a virtual environment and pin numpy/pandas versions.
2. Run the generator to a temporary output path with the same config and seed; do not overwrite supplied files. Compare hashes to see if these CSVs match the checked-in generator.
3. Record command/config, seed, SHA-256 hashes, row counts and date in data/manifest.json.
4. Create docs/data_profile.md with shapes, nulls, categories, attribute distributions, event counts by day/scope, degree distribution, isolates, and cross-constituency edges.
5. Never hand-edit generated CSVs; create new data only through a reproducible config/command.

### 2.2 Constituency loader
Use constituency 1 (C01) for default smoke tests; Step 5 selected C21 as the closer fictional race for the planned campaign demo. Neither is representative of a real electorate.

- Select voters/candidates by constituency ID, not row order.
- Load events scoped to C01, its region, and state (empty region and constituency IDs mean state-level).
- Core graph: keep only edges with both source and target in C01. Exclude and report boundary edges; stretch 1 can re-enable them through a spillover coefficient.
- An edge source -> target means the source influences the target. Construct incoming adjacency by target ID.
- Validate unique IDs, resolved references, ranges/categories, no self/duplicate edges, target geography in edge metadata, event dates/scopes, and no accidental voter mixing.

Generator validation is useful but not enough; independently validate on load. Keep transformations scripted and add a data dictionary.

### 2.3 Data meaning
See data/README.md and data/USAGE.md.

| Column | Model meaning |
|---|---|
| initial_opinion | Starting scalar opinion in [-1,1]. |
| susceptibility | Voter movement under social influence. |
| influence_power | Sender influence along outgoing edges; distinct from susceptibility. |
| confidence_threshold | Maximum opinion distance for accepting influence; generated in [0,2]. |
| turnout_propensity | Baseline probability-like turnout tendency. |
| event_sensitivity | Individual response to events. |
| issue_* | Issue importance; normalize to sum to 1 for weighted calculations. |
| candidate position_* | Candidate stance on each issue in [-1,1]. |
| local_popularity | Synthetic candidate appeal prior. |
| edge_weight | Directed tie strength. |
| event intensity/credibility/duration/target_group | Signal strength, believability, active time, intended audience. |
| candidate_effect | Synthetic signed signal for named candidate; not measured causal effect; blank means no candidate named. |
| issue_direction | Optional signed direction for directional, neutral issue-specific events in [-1, 1]; older schedules may use opinion_direction. |

There is no truth/misinformation label, campaign reach, voter choice, or official result in the supplied CSVs. Scenario actions live separately under `campaign_actions/` and are model controls, not observations. The implemented message labels are `ordinary`, `misleading`, and `smear`; the smear label represents a fabricated claim about another candidate. Don't mutate `candidate_effect` to manufacture factuality.

For a real benchmark, create data/real_benchmark/ with official candidate results, registered electorate, ballots cast, turnout definition, date, source URL/retrieval date/terms, candidate mappings and cleaning script. Keep it separate. Never initialize agents from final results and then call the match a prediction.

## 3. Model specification

The core model is implemented; `docs/decisions.md` and `docs/model.md` remain to be written before the final report. They should fix the constituency/demo choice, candidate scoring, campaign/event semantics, update order, turnout/vote rules, parameter defaults, seeds, and the exact scope of any empirical claim. Suggested parameter ranges are assumptions for sensitivity analysis, not values inferred from papers.

### 3.1 Voter state and candidate support
Each voter has scalar opinion x_i(t) in [-1,1], supplied attributes, then a turnout and vote. The default scalar candidate utility is:
$U_ic = b_pop * logit(clamp(local_popularity_c)) - b_ideol * abs(x_i - ideology_c)$.
Define ideology_c as the mean of the candidate's five issue positions; this compresses a multi-issue platform to one axis. An opt-in `synthetic_issue_profile` sensitivity mode instead uses:
$U_ic = b_pop * logit(clamp(local_popularity_c)) - b_ideol * sum_k(w_ik * abs(q_ik(t) - p_ck))$.
Here w_ik is voter i's normalized importance for issue k, p_ck is candidate c's supplied position, and q_ik(t) is a synthetic voter issue position. Since the data has no voter issue positions, q is initialized reproducibly from the scalar opinion plus seeded, zero-mean issue offsets, clipped to [-1,1]; daily scalar opinion changes shift all five issue preferences. `synthetic_issue_dispersion` controls the maximum offset (0 to 1). These positions are scenario assumptions, not measured beliefs. Keep the scalar mode as the default/control; replace the synthetic generator if measured voter issue positions become available. Issue-specific neutral events update the corresponding q dimension in issue-aware mode; scalar mode remains compatible. Convert utilities to support probabilities with softmax temperature tau (lower tau -> more deterministic). Keep every constituency candidate; don't infer ideology from party names. Check mappings with hand examples. See `ISSUE_PREFERENCE_STATE.md` for the implemented state rules and limitations.

### 3.2 Social update: DeGroot + bounded confidence
For target i, find incoming neighbors j -> i. Include j only when abs(x_j-x_i) <= confidence_threshold_i. Weight senders by edge_weight * influence_power_j^rho; normalize over eligible senders. If none qualify, social mean m_i = x_i. Synchronous update:
x_i_new = clip(x_i + susceptibility_i * lambda_social * (m_i-x_i), -1, 1).
Every new opinion reads the complete day-t snapshot; no CSV-order dependence. Keep directed edges; don't silently add self-loops.

Initial sweeps:
- lambda_social: 0.05, 0.10, 0.20, 0.40, 0.60.
- Confidence multiplier on supplied voter thresholds: 0.5, 0.75, 1, 1.25 (clip [0,2]); controlled global values: 0.1, 0.25, 0.5, 1, 1.5.
- Sender exponent rho: 0, 0.5, 1.
- Susceptibility multiplier: 0.5, 1, 1.5 (clip to [0,1]).
Log opinion distribution, mean, SD, quantiles, and eligible edges each day.

### 3.3 Candidate and manipulative campaign agents

**Current implementation scope:** actions support `ordinary` platform persuasion, `misleading` platform-position distortion, and `smear` claims against another candidate. `smear` is the fabricated-claim mechanism: it changes belief/target utility and may spread socially. There is no separate `fabricated` message-position type; the explicit truth/mechanics labels are `ordinary`, `misleading`, and `smear`. Keep these meanings distinct in experiments and reports.
Actions run on 30-day timeline. Keep intent/truth label distinct from voter response:
1. none: no campaign.
2. ordinary: pull toward candidate's declared platform/policy signal.
3. misleading: same mechanics but signal position is deliberately distorted.
4. smear: a fabricated claim about another candidate; recipients may believe it, and believers receive a utility penalty for that target. Rumor spread and fact-check response are separate mechanisms.

For reached voter i, candidate c, signal q, reach indicator r_i, intensity a, susceptibility s_i:
x_i_campaign = clip(x_i + r_i*a*s_i*lambda_campaign*(q-x_i), -1, 1).
If also changing candidate utility directly, specify a separate effect and prevent double counting. Log action, timing, target, reach, signal, truth status and seed.

Map groups explicitly: Rural -> urban_rural=Rural; Students -> occupation=Student; Farmers -> Farmer or Agricultural labourer. Define remaining mappings; fail on unknown labels. Intended target is not evidence of actual exposure.

Sweep lambda_campaign {0.05,0.1,0.2,0.4}, reach {0.1,0.25,0.5,1}, distortion {0.1,0.25,0.5,1} on [-1,1], frequency {0,1,3,7} days per action window. Always run no-campaign control.

### 3.4 Shared adversarial targeting
Policies: uniform random; explicit target group; high susceptibility; high influence (weighted out-degree combined with influence_power); and seeded community targeting. Community labels are inferred once per run using seeded label propagation on the weak projection of the social graph; these are model-derived clusters, not observed geographic or social identities. Current group mappings: Rural/Urban -> urban_rural, Students -> occupation=Student, Farmers -> Farmer or Agricultural labourer, Unemployed -> occupation=Unemployed, and income groups -> wealth_level. Keep the constituency-wide requested exposure budget fixed across policies. If a group or selected community is smaller than the requested budget, select its full eligible audience and report the shortfall. High-susceptibility ranking uses effective susceptibility; high-influence ranking uses weighted out-degree multiplied by influence_power. Compare matched timing, intensity, signal and seeds so targeting is the only changed factor. Record requested/selected counts, coverage, audience attribute summaries and impact. Keep synthetic recipient IDs internal, not in public report. Treat efficiency as simulated behavior under assumptions, not real campaign advice or causal evidence.

### 3.5 News/event agent and shock scheduler
The simulator now filters constituency, matching-region and state-wide records, activates events on `event_day` through `event_day + duration - 1`, and maps target groups to voter attributes. For voter i, the per-active-day signal is:
`lambda_event * intensity_e * credibility_e * event_sensitivity_i * normalized_issue_weight_ik / duration_e` (issue weight is 1 for general).

Candidate-linked events adjust the named local candidate's utility by signed `candidate_effect`; candidate-free events perturb opinion using a seeded, zero-mean direction. An event naming a candidate outside the local ballot does not boost a local candidate and is treated as neutral for this constituency. Signals accumulate across active days, then decay using a configurable half-life (default 7 days); optional reminders can reinforce them. A zero half-life makes effects persistent. Optional schedule fields support probabilistic occurrence, dependencies on an earlier event, and structural candidate withdrawal. `lambda_event=0` is the no-shock control; default strength is 0.2. Per-day applications, decay, reminders and exposure totals are logged. Duration 1 applies once.

`compare_event_shock` runs one matched shock/no-shock pair; `run_event_seed_sweep` repeats that pair for an explicit seed list. `run_event_factorial_sweep` compares event arms against same-seed controls. `run_batch` and `compare_campaign` now provide the broader matched-seed experiment workflow, including candidate-level tidy outcomes; `summarize_batch` reports medians and empirical 5th–95th percentile ranges. `build_step5_scenarios` supplies frozen/social controls, campaign/news combinations, candidate-agent strategy arms, and equal-reach targeting comparisons. `election_sim.step5_cli` scans the constituency baselines, selects the closest support race, runs the 30-seed scenario matrix, and evaluates regional spillover at 0, 0.05, 0.1, and 0.2. These are synthetic scenario results, not empirical uncertainty intervals. Candidate-free events support balanced seeded shocks or explicitly directional shifts along the scalar opinion axis. Still to implement: mappings from directional shocks to named political responses (the current voter/candidate data has no validated “radical-change” or “nationalist” response attribute). Generator event counts are scenario assumptions, not event-rate estimates.

### 3.6 Turnout and election result
Turnout samples Bernoulli(p_turnout) at election close. With default-zero coefficients this preserves the supplied propensity; optional exposure terms are:
p_turnout_i = sigmoid(logit(clamp(turnout_propensity_i)) + gamma_mob*campaign_exposure_voter_days_i + gamma_event*news_exposure_voter_days_i).
`gamma_mob` and `gamma_event` default to zero; do not fit against synthetic pseudo-truth. Participating voters sample from candidate softmax support with a dedicated vote RNG; compare argmax deterministically as sensitivity. Abstainers cast zero. Verify votes + abstentions = population, one vote per voter, candidate totals = turnout, shares sum to 1 when votes exist.

### 3.7 Daily order
For each day: snapshot state; resolve today's actions/events/audiences; compute social signals from the snapshot and update synchronously; apply campaign and active event signals; update cumulative voter-day exposure totals; recompute support; save daily aggregates and sampled view state. At election close, convert final cumulative exposure to turnout probability, then sample turnout and votes. This order is a modeling choice: document it and optionally compare an alternative. Derive RNG child streams from the master seed for targeting, shocks, turnout, voting, and event directions; never rely on global uncontrolled randomness.

## 4. Software architecture and verification
Suggested modules: config; data loader/validators; network/adjacency; voter/support; candidate actions; shared targeting; events; bounded-confidence dynamics; turnout/vote; simulation loop; experiments/manifests; metrics; plots/dashboard. Keep UI separate from model.

Expose simulate(config,data,seed) -> RunResult with daily summaries, candidate results, turnout, diagnostics and metadata. Every run saves resolved config, master/child seeds, data hashes, code revision, metrics and timestamp. Pin dependencies and provide one command for baseline and one for report figures.

Use small deterministic fixtures to verify:
- Opinions finite/in [-1,1]; support probabilities sum to 1.
- Zero influence leaves state unchanged; no eligible neighbors means no social movement.
- One directed edge changes only its target as formula predicts.
- Zero campaign reach changes nobody; full reach selects all eligible voters.
- Event scopes/targets/dates/durations, memory, reminders, dependencies and structural withdrawal work; duration 1 applies once.
- In issue-aware mode, a neutral issue-specific event changes only that voter preference dimension; general shocks retain the scalar path; scalar mode preserves compatibility.
- Neutral no-candidate news does not directly alter candidate utility.
- Turnout 0 -> no votes; turnout 1 -> one ballot per voter.
- Fixed config/data/seed reproduces exactly; vote accounting reconciles.
- Paired shock/no-shock runs share random streams.
Generator tests do not substitute for simulation verification.

## 5. Experiment plan and evaluation

### 5.1 Scenario matrix
Pilot with 3-5 seeds during debugging; use at least 30 seeds for final reports:
1. Frozen opinions, no campaign/events.
2. Social influence only.
3. Social + ordinary uniform campaign.
4. Social + misleading or smear campaign.
5. Social + neutral news/external events.
6. Social + campaign + events.
7. Equal-budget targeting comparison.
8. Paired counterfactuals with identical graph, initialization and random streams: manipulation on/off; shock on/off; ordinary/misleading with same reach/timing/intensity.

### 5.2 Metrics
Report medians/means and 5th-95th seed percentiles:
- Candidate vote count/share, turnout/rate, winner/margin.
- Daily support, opinion mean/SD/quantiles, opinion movement, final polarization (define variance and/or clusters plus threshold).
- Paired manipulation effects on target and electorate support, vote share, winner/margin, opinion movement.
- Paired shock effects and fraction with absolute opinion change >0.05 and >0.10; affected days and optional graph-distance spread.
- Targeting impact under equal exposure; simulated efficiency under assumptions.
- Sensitivity to seed, confidence/social/campaign parameters, event strength/credibility/reach, targeting.

Step 5 is implemented in `election_sim/experiments.py` and runnable with
`python -m election_sim.step5_cli`. The runner provides `run_batch`,
`compare_campaign`, `summarize_batch`, and a named scenario matrix; it scans
all constituency baselines, runs the selected close race, and evaluates the
regional spillover sweep with matched seeds. The completed 30-seed run is in
`results/step5_seed0_29/`: seed-0 screening selected C21 with a 0.0433
top-two mean-support gap, and the final matrix uses seeds 0–29. These are
synthetic scenario comparisons; their percentile ranges are not estimates of
real-world uncertainty. Agent arms share a contact-budget cap and targeting
arms request equal reach, but the saved batch tables omit per-seed agent
budget-use and audience-shortfall diagnostics; inspect those diagnostics
before making a strict realized-contact comparison.

### 5.3 Issue-aware support with candidate agents
The support-sensitivity runner accepts the same candidate-agent definitions in
each matched arm. For each seed, compare scalar support with synthetic
issue-profile support at configured dispersions while keeping agent settings,
event conditions, and shared random streams fixed. Save per-seed outcomes and
agent poll/action records; summarize changes in agent choices and poll shares
against the scalar arm. The zero-dispersion issue-aware arm remains distinct
from scalar support because it still scores each candidate's issue positions
separately. This is a simulator sensitivity analysis, not evidence of how real
campaigns or voters behave. The five-seed pilot is in
`results/agent_issue_support_pilot_5seeds`; the 30-seed C01 run is in
`results/c01_agent_support_sensitivity_30seed` (180 matched arms, 3,780 logged
agent decisions). In this scenario, winner and turnout-rate changes were zero;
mean absolute support difference versus scalar increased from 0.0173 at
dispersion 0 to 0.0477 at dispersion 1, while agent-action change rate rose
from 12.7% to 23.0%. These are model-specific sensitivity results, not
real-world estimates.

Example:

```powershell
python -m election_sim --config configs/agents.json --agents campaign_agents/c01_agents.json `
  --support-sensitivity --support-seed-count 30 --support-dispersions 0 0.2 0.35 0.6 1 `
  --seed 0 --output results/c01_agent_support_sensitivity
```

### 5.4 Optional real-election comparison
No real labels exist in current repo. Obtain official completed-election candidate totals and registered electorate/ballots cast; prefer pre-election polling/survey to constrain initial opinions. Do not use final result to initialize or tune. Where possible freeze a holdout before tuning. Report vote-share MAE/RMSE, turnout error in percentage points with matched denominator, winner accuracy and seed intervals. Compare to persistence from pre-election polling if available. Validate polarization only against independent opinion data. Without independent pre-election inputs, label any result match aggregate reconstruction/external reference, not predictive validation.

## 6. Dashboard and report
Show:
- Constituency (C01 default), scenario, seed controls.
- Candidate support/vote-share by day with seed interval and campaign/event markers.
- Opinion distribution and defined polarization.
- Small sampled network colored by opinion/leading candidate; avoid plotting all 3,000 nodes/edges by default.
- Campaign details (candidate, ordinary/misleading/smear, target, reach, intensity, distortion); event details (source, issue, scope, day, duration, intensity, credibility).
- Final turnout/result; distinguish simulated from observed and show official benchmark only if obtained.
- Persistent visible warning that demographics, networks, events and behavioral values are synthetic.
Exports include config, data hashes and seed. Use stable candidate colors; UI choices don't silently change baseline.

## 7. Eight-week schedule and team
| Week | Work | Exit gate |
|---|---|---|
| 1 | Confirm C01, definitions, update order, turnout, claim, data profile, interfaces. | decisions/model draft/config |
| 2 | Frozen data, loader, scope filtering, graph diagnostics. | valid C01 slice; no boundary contamination |
| 3 | Voter state, candidate utility, social dynamics, daily loop. | hand examples and seeded baseline |
| 4 | Campaign/action schema, manipulation, targeting. | paired scenarios and impact metrics |
| 5 | News/event scheduler, neutral shocks, turnout/vote. | scope, duration and accounting verified |
| 6 | Experiment runner, seed batches, sweeps, sensitivity, figures. | reproducible metrics/manifests |
| 7 | Obtain benchmark if feasible; dashboard/report integration. | benchmark or explicit limitation |
| 8 | Clean environment reproduction, report, slides/demo; stretch only if core complete. | README reproduces headline run |

Roles:
- A Voter & Network Agents: directed graph, voter state, bounded-confidence baseline.
- B Malicious Candidate Agents: action schema, ordinary/misleading/smear signals.
- C News/Event Agents: event scope/time, neutral shocks, candidate-associated effects.
- D Adversarial Targeting: recipient policies, equal-budget tests, manipulation metrics.
- E Verification & Testing: integration, benchmark sourcing, reproducibility, dashboard/report.
Agree interfaces Week 1; integrate continuously from Week 3; another member reviews each module's assumptions.

## 8. Regional implementation and remaining stretch
**Implemented multi-constituency region:** use cross-constituency links only with an explicit spillover coefficient. IDs are consecutive synthetic labels, not geographic adjacency evidence. Compare 0, 0.05, 0.1, 0.2 and report sensitivity. Reference [4] motivates hybrid dynamics, not empirical spillover.

**Multi-phase national:** define phases; let early synthetic results affect later campaign through an explicit bandwagon/discouragement parameter. Compare feedback on/off with paired seeds. Ref [3] motivates damage-spread analysis, not national prediction.

Suggested outputs: docs/decisions.md, docs/model.md, docs/data_profile.md, docs/data_dictionary.md; data/manifest.json, separate campaign_actions and optional real_benchmark; configs/; src/election_sim/ modules; tests/; results/<run_id>/ manifests/metrics/figures; reports/; README with setup, baseline, experiments, dashboard and reproduction commands.

## 9. Acceptance checklist
- [x] C01 runs from supplied files without hand edits and is labeled synthetic.
- [x] Social update is directed, bounded-confidence, synchronous and seed-reproducible.
- [x] Campaigns have explicit type/truth label, target, reach, timing and rule.
- [x] Events honor scope/date/duration/target/issue/intensity/credibility/sensitivity; neutral events don't endorse candidates.
- [x] Equal-reach targeting, paired campaign/shock comparisons, and multi-seed summaries are available.
- [x] Turnout/votes reconcile; batch results report seed summaries.
- [x] Step 5 baseline scan, close-race selection, 30-seed matrix, and regional spillover sweep are recorded in `results/step5_seed0_29/`.
- [ ] Dashboard, outputs and documented commands reproduce headline runs.
- [ ] Real-election claims, if any, use separately sourced official data and avoid final-result leakage.
- [ ] Report separates code verification, synthetic scenario behavior and empirical validation.
- [ ] Stretch starts after core acceptance.

## 10. Reference guide
Supplied PDFs support mechanisms, not this generator's assumptions or parameter values.

1. Fortunato et al., “Monte Carlo Simulations of Opinion Dynamics” - bounded-confidence background. references/0501105v1.pdf; [arXiv](https://arxiv.org/abs/cond-mat/0501105).
2. Charcon & Monteiro, “A multi-agent system to predict the outcome of a two-round election” - election modeling example; its two-round setting differs. references/1-s2.0-S0096300320304409-main.pdf.
3. Fortunato, “Damage Spreading and Opinion Dynamics on Scale Free Networks” - paired shock propagation motivation; thresholds are paper-specific. references/0405083v1.pdf; [arXiv](https://arxiv.org/abs/cond-mat/0405083).
4. Xue & Roy, “Averager-copier-voter models for hybrid opinion dynamics in complex networks” - hybrid consensus background. references/1903.12329v1.pdf; [arXiv](https://arxiv.org/abs/1903.12329).
5. Saburov & Saburov, “Reaching a Nonlinear Consensus: Polynomial Stochastic Operators” - nonlinear consensus theory, not election/campaign specification. references/s12555-014-0061-0.pdf; [DOI](https://doi.org/10.1007/s12555-014-0061-0).

