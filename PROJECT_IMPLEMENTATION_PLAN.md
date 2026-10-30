# Election Dynamics Simulator - Complete Implementation Plan

## 1. Goal and scope

Implement the updated proposal's virtual election: voters start with synthetic opinions/attributes, influence one another through a directed social graph, respond to candidate campaigns including explicitly labeled misleading/fabricated signals, and experience neutral but disruptive news/external events. A turnout rule converts final preferences into votes. Core: one constituency, fully reproducible. Multi-constituency/state and national layers are stretch work.

**Data status:** The repo has a synthetic data generator and fictional state, not real election results. Supplied data: 3 regions, 21 constituencies, 3,000 voters each (63,000 total), 3 candidates each (63), 506,629 directed edges, and 278 events over 30 days. All values are generated assumptions, not facts about real voters. Use them for mechanics and scenario demonstrations. If the final report compares to a real election, obtain official results separately. Without independent pre-election opinion data, call results a synthetic scenario or aggregate reconstruction, not a validated forecast.

Core outputs: single-constituency simulation; DeGroot/bounded-confidence dynamics; candidate agents with ordinary and manipulative campaigns; news/event agents; shared targeting policies; turnout/results; experiment runner; dashboard/figures; report and demo.

Stretch only after core: (1) multiple constituencies with explicit spillover, (2) synthetic multi-phase national layer where early results influence later campaigns.

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
Use constituency 1 (C01) as the default demo: it is just a convenient fictional slice, not representative of a real electorate.

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

There is no truth/misinformation label, campaign reach, voter choice, or official result in the CSVs. Add separate scenario actions in data/campaign_actions/ as JSON/CSV with action/candidate ID, start day, duration, signal type (policy, persuasion, misleading, fabricated), issue, target group/policy, intensity, reach fraction, message position/distortion, truth-status label, rationale. These are model controls, not observations. Don't mutate candidate_effect to manufacture factuality.

For a real benchmark, create data/real_benchmark/ with official candidate results, registered electorate, ballots cast, turnout definition, date, source URL/retrieval date/terms, candidate mappings and cleaning script. Keep it separate. Never initialize agents from final results and then call the match a prediction.

## 3. Model specification

Create docs/decisions.md and docs/model.md before implementation. Fix the constituency, candidate scoring, campaign/event semantics, update order, turnout/vote rules, parameter defaults, seeds, and exact empirical claim. Suggested parameter ranges are assumptions for sensitivity analysis, not values inferred from papers.

### 3.1 Voter state and candidate support
Each voter has scalar opinion x_i(t) in [-1,1], supplied attributes, then a turnout and vote. First candidate utility:
$U_ic = b_pop * logit(clamp(local_popularity_c)) - b_ideol * abs(x_i - ideology_c)$.
Define ideology_c as the mean of the candidate's five issue positions; document that this compresses a multi-issue platform to one axis. Convert utilities to support probabilities with softmax temperature tau (lower tau -> more deterministic). Later add issue-aware fit using normalized issue importance and candidate issue positions. Keep every constituency candidate; don't infer ideology from party names. Check mapping with hand examples.

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
Actions run on 30-day timeline. Keep intent/truth label distinct from voter response:
1. none: no campaign.
2. ordinary: pull toward candidate's declared platform/policy signal.
3. misleading: same mechanics but signal position is deliberately distorted.
4. fabricated: candidate-associated false event-like signal, explicitly labeled. Don't claim fact checking unless implemented.

For reached voter i, candidate c, signal q, reach indicator r_i, intensity a, susceptibility s_i:
x_i_campaign = clip(x_i + r_i*a*s_i*lambda_campaign*(q-x_i), -1, 1).
If also changing candidate utility directly, specify a separate effect and prevent double counting. Log action, timing, target, reach, signal, truth status and seed.

Map groups explicitly: Rural -> urban_rural=Rural; Students -> occupation=Student; Farmers -> Farmer or Agricultural labourer. Define remaining mappings; fail on unknown labels. Intended target is not evidence of actual exposure.

Sweep lambda_campaign {0.05,0.1,0.2,0.4}, reach {0.1,0.25,0.5,1}, distortion {0.1,0.25,0.5,1} on [-1,1], frequency {0,1,3,7} days per action window. Always run no-campaign control.

### 3.4 Shared adversarial targeting
Policies: uniform random; explicit target group; high susceptibility; high influence (weighted out-degree/influence_power); community (only with documented seeded assignment). Compare equal exposure budgets, timing, intensity and signal so only targeting changes. Record counts, coverage, attribute summaries and impact. Keep synthetic recipient IDs internal, not in public report. Treat efficiency as simulated behavior under assumptions, not real campaign advice or causal evidence.

### 3.5 News/event agent and shock scheduler
Resolve C01, region and state scopes. Event e is active from event_day through event_day+duration-1. For voter i, event strength:
h_ie = intensity_e * credibility_e * event_sensitivity_i * normalized_issue_weight_ik (use 1 for general).
Candidate-associated events may change support for that named candidate using candidate_effect. News/External with no candidate must not directly favor one. For neutral events, perturb issue opinion using a documented zero-mean seeded direction; direction is a scenario variable, not inferred bias.

Recommended timing: apply shock at onset, then linearly decay residual through duration; log daily effects and ensure duration 1 applies once. Controls: no events, supplied events, zero-strength neutral events, mild and high shock. Use common random numbers for paired shock/no-shock comparisons.

### 3.6 Turnout and election result
First version samples Bernoulli(turnout_propensity) at election close. Later sensitivity model:
p_turnout_i = sigmoid(logit(clamp(turnout_propensity_i)) + gamma_mob*campaign_mobilization + gamma_event*event_mobilization).
Start gamma values at zero; don't fit against synthetic pseudo-truth. Participating voter samples from candidate softmax support with dedicated vote RNG; compare argmax deterministically as sensitivity. Abstainers cast zero. Verify votes + abstentions = population, one vote per voter, candidate totals = turnout, shares sum to 1 when votes exist.

### 3.7 Daily order
For each of 30 days: snapshot state; resolve today's actions/events/audiences; compute social signals from snapshot and update synchronously; apply campaign signal; apply event shock/decay; recompute support and optional mobilization; save daily aggregates and sampled view state. At close sample turnout/vote and aggregate. This order is a modeling choice: document it and optionally compare an alternative. Derive RNG child streams from master seed for targeting, shocks, turnout and voting; never rely on global uncontrolled randomness.

## 4. Software architecture and verification
Suggested modules: config; data loader/validators; network/adjacency; voter/support; candidate actions; shared targeting; events; bounded-confidence dynamics; turnout/vote; simulation loop; experiments/manifests; metrics; plots/dashboard. Keep UI separate from model.

Expose simulate(config,data,seed) -> RunResult with daily summaries, candidate results, turnout, diagnostics and metadata. Every run saves resolved config, master/child seeds, data hashes, code revision, metrics and timestamp. Pin dependencies and provide one command for baseline and one for report figures.

Use small deterministic fixtures to verify:
- Opinions finite/in [-1,1]; support probabilities sum to 1.
- Zero influence leaves state unchanged; no eligible neighbors means no social movement.
- One directed edge changes only its target as formula predicts.
- Zero campaign reach changes nobody; full reach selects all eligible voters.
- Event scopes/targets/dates/durations work; duration 1 applies once.
- Neutral no-candidate news doesn't favor a candidate.
- Turnout 0 -> no votes; turnout 1 -> one ballot per voter.
- Fixed config/data/seed reproduces exactly; vote accounting reconciles.
Generator tests do not substitute for simulation verification.

## 5. Experiment plan and evaluation

### 5.1 Scenario matrix
Pilot with 3-5 seeds during debugging; use at least 30 seeds for final reports:
1. Frozen opinions, no campaign/events.
2. Social influence only.
3. Social + ordinary uniform campaign.
4. Social + misleading/fabricated campaign.
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

### 5.3 Optional real-election comparison
No real labels exist in current repo. Obtain official completed-election candidate totals and registered electorate/ballots cast; prefer pre-election polling/survey to constrain initial opinions. Do not use final result to initialize or tune. Where possible freeze a holdout before tuning. Report vote-share MAE/RMSE, turnout error in percentage points with matched denominator, winner accuracy and seed intervals. Compare to persistence from pre-election polling if available. Validate polarization only against independent opinion data. Without independent pre-election inputs, label any result match aggregate reconstruction/external reference, not predictive validation.

## 6. Dashboard and report
Show:
- Constituency (C01 default), scenario, seed controls.
- Candidate support/vote-share by day with seed interval and campaign/event markers.
- Opinion distribution and defined polarization.
- Small sampled network colored by opinion/leading candidate; avoid plotting all 3,000 nodes/edges by default.
- Campaign details (candidate, ordinary/misleading/fabricated, target, reach, intensity, distortion); event details (source, issue, scope, day, duration, intensity, credibility).
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
- B Malicious Candidate Agents: action schema, ordinary/misleading/fabricated signals.
- C News/Event Agents: event scope/time, neutral shocks, candidate-associated effects.
- D Adversarial Targeting: recipient policies, equal-budget tests, manipulation metrics.
- E Verification & Testing: integration, benchmark sourcing, reproducibility, dashboard/report.
Agree interfaces Week 1; integrate continuously from Week 3; another member reviews each module's assumptions.

## 8. Stretch and repository outputs
**Multi-constituency:** use cross-constituency links only with explicit spillover coefficient. IDs are consecutive synthetic labels, not geographic adjacency evidence. Compare 0, 0.05, 0.1, 0.2 and report sensitivity. Reference [4] motivates hybrid dynamics, not empirical spillover.

**Multi-phase national:** define phases; let early synthetic results affect later campaign through an explicit bandwagon/discouragement parameter. Compare feedback on/off with paired seeds. Ref [3] motivates damage-spread analysis, not national prediction.

Suggested outputs: docs/decisions.md, docs/model.md, docs/data_profile.md, docs/data_dictionary.md; data/manifest.json, separate campaign_actions and optional real_benchmark; configs/; src/election_sim/ modules; tests/; results/<run_id>/ manifests/metrics/figures; reports/; README with setup, baseline, experiments, dashboard and reproduction commands.

## 9. Acceptance checklist
- [ ] C01 runs from supplied files without hand edits and is labeled synthetic.
- [ ] Social update is directed, bounded-confidence, synchronous and seed-reproducible.
- [ ] Campaigns have explicit type/truth label, target, reach, timing and rule.
- [ ] Events honor scope/date/duration/target/issue/intensity/credibility/sensitivity; neutral events don't endorse candidates.
- [ ] Equal-budget targeting and paired manipulation/shock metrics are available.
- [ ] Turnout/votes reconcile; results include seed uncertainty.
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

