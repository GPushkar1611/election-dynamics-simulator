# Election Dynamics Simulator: End-to-End Project Plan

## 1. Purpose and scope

Build a reproducible, single-constituency agent-based simulation of an election campaign. Each voter is an agent with a latent preference over candidates, a social-neighbor network, and a probability of turning out. Candidate/party agents issue campaign signals; scheduled news events can perturb preferences. At the end of the campaign, simulated turnout and votes produce a constituency result. Calibrate and evaluate aggregate results against one real election.

The committed deliverable is the single-constituency model, its data and calibration record, reproducible experiments, visualizations, and final report/demo. Multi-constituency spillovers and a multi-phase national election are optional stretch work and must not delay the core.

This is a research simulation, not an individual-level prediction system. Aggregate results can validate aggregate vote shares and turnout, but cannot establish that the simulated individual opinion trajectories are real. Do not claim causal campaign effects from one election's aggregate outcome.

## 2. Decisions to record before implementation

Create `docs/decisions.md` (or an equivalent project decision log) and record these choices with date, rationale, and owner:

1. **Election and constituency.** Select a completed, single-round election with official polling-station or constituency results and a reliable total electorate/turnout denominator. Prefer a place where candidate/party identities and the election date are unambiguous. Use one election for the main holdout evaluation; if multiple historical elections are available, reserve them for robustness checks rather than quietly mixing them into calibration.
2. **Unit of analysis.** Use polling stations/precincts as aggregate rows when available; otherwise use the constituency total. Never treat an aggregate row as an individual voter.
3. **Candidate scope.** Include all candidates/parties with reported official vote totals. If simplifying to the leading candidates plus `Other`, document the mapping and preserve the exact reported totals for final comparison.
4. **Data policy.** Use public, lawfully reusable, documented data. Do not collect or store names, voter IDs, addresses, or other personally identifying data. Store source URLs, retrieval date, license/terms, raw-file checksum, and transformations.
5. **Implementation stack.** Choose one language and dependency manager for the simulation, dashboard, and figures; pin versions. Keep simulation logic separate from UI code.

Do not invent demographic or social-network observations. If no local individual survey/network data is available, label voter traits and ties as synthetic assumptions and run sensitivity analyses over them.

## 3. Data plan

### 3.1 Required data

Create a versioned, documented data directory with:

- Official final vote totals by candidate/party for the chosen constituency and election.
- Registered electorate and total ballots cast/turnout, using the official denominator definition.
- Candidate/party names, party affiliation, and any official alliances relevant to interpreting the result.
- Election date and, if available, campaign start/end dates.
- Optional polling-station/precinct results, if the team wants to validate sub-constituency variation.
- Optional publicly available, suitably licensed polling or survey time series for candidate preference. If unavailable, the model's initial preferences are assumptions, not measured baseline opinion.
- Optional public event/campaign timeline. Each event needs a date, source, affected issue/candidate, and a stated mapping to a simulated signal. Do not turn a headline into a numerical effect without declaring the assumption.

### 3.2 Source and cleaning procedure

1. Find the election management body's official result publication first. Record URL, publication name, retrieval date, license/terms, and downloaded-file checksum in `data/SOURCES.md`.
2. Prefer machine-readable official CSV/XLSX/JSON. If only PDF is available, extract it and manually spot-check totals against the source; retain the original PDF.
3. Normalize candidate names and party labels through an explicit mapping table; retain source labels in a separate column.
4. Check that candidate vote counts are nonnegative integers, candidate names are unique after mapping, and the sum of candidate votes matches the official valid-vote total (with any blank/invalid ballot categories handled explicitly).
5. Check turnout arithmetic against the source's stated denominator. Document whether turnout means ballots cast / registered electors and whether postal/absentee votes are included.
6. For precinct data, check duplicate precinct IDs, missing precincts, subtotal-vs-constituency reconciliation, and boundary or candidate changes. Do not silently discard rows.
7. Save immutable raw inputs under `data/raw/`, cleaned inputs under `data/processed/`, and a script/notebook that recreates processed data. Never hand-edit the cleaned file without encoding the change in the transformation script.
8. Write a data dictionary and a short limitations note. If demographic or survey data are not available, say so and keep those variables synthetic.

### 3.3 Proposed data schema

- `election.json`: election ID, date, jurisdiction, constituency ID/name, result source, denominator definition, candidate list, data version.
- `results.csv`: `constituency_id,candidate_id,party_id,votes,valid_votes,registered_electors,ballots_cast` (one row per candidate; election metadata may be normalized into a separate file).
- Optional `precinct_results.csv`: `precinct_id,candidate_id,votes,valid_votes,registered_electors,ballots_cast`.
- Optional `polls.csv`: `date,candidate_id,share,sample_size,source,method`; record uncertainty/method where supplied.
- `events.csv`: `event_id,date,description,source_url,affected_candidates_or_issues,signal_vector,assumption_note`.

## 4. Model specification

Document equations, parameter meanings, update order, tie-breaking, random-number handling, and defaults in `docs/model.md` before calibration.

### 4.1 Voter agents and opinions

Represent voter `i` by a candidate-support vector `x_i(t)` of length `K`, where each value is nonnegative and the values sum to 1. This supports multi-candidate elections and avoids pretending one scalar opinion captures all candidates. A simpler scalar bounded-confidence implementation is acceptable only for a documented two-candidate pilot; the final core model should support the actual candidate set or a justified `Other` category.

Initialize preferences from one of these sources, in order of preference:

1. Public pre-election survey/poll microdata with suitable candidate preference measures and documented sampling limitations.
2. A documented distribution constrained by polls or other public aggregate evidence.
3. A synthetic prior whose mean is stated as an assumption and varied in sensitivity analysis.

Do not initialize each voter from final election results and then claim the simulation predicted those same results. If final results are used as calibration targets, keep them out of initial conditions and clearly describe calibration vs evaluation.

### 4.2 Social network

Build a graph with one node per simulated voter. With observed social-network data, use that graph only if its provenance and coverage support the selected population. Otherwise use a clearly labeled synthetic graph. Recommended baseline: undirected Watts-Strogatz small-world graph, initialized with mean degree `k=8` and rewiring probability `p=0.1`; compare with Erdős-Rényi and Barabási-Albert alternatives as structural sensitivity checks. These are starting assumptions, not empirical facts about the electorate.

Keep the graph fixed for the core simulation. Normalize each voter's neighbor weights to sum to 1; optionally include a self-weight `w_self` (baseline 0.2, tune 0-0.5). If a voter has no social neighbors, use the voter's own state plus campaign/event signals only.

### 4.3 Opinion update rule

Use a DeGroot-style weighted update gated by bounded confidence. For voter `i`, form the eligible neighbor set `N_i(t) = {j in neighbors(i): distance(x_i(t), x_j(t)) <= epsilon}`. Use total-variation distance `0.5 * sum_c |x_ic - x_jc|` for candidate-support vectors. The social signal is the normalized weighted average of eligible neighbors' preferences; if none qualify, retain the prior opinion for the social component.

Combine social interaction, campaigns, and news with an explicitly normalized convex update:

`z_i(t) = (1 - alpha_campaign - alpha_event) * ((1 - alpha_social) * x_i(t) + alpha_social * social_i(t)) + alpha_campaign * campaign_i(t) + alpha_event * event_i(t)`

Then project `z_i(t)` onto the probability simplex to ensure nonnegative values summing to 1. Clarify whether `alpha_social` is applied only when eligible neighbors exist. Implement synchronous updates from a snapshot of all opinions at time `t` for the baseline, so results do not depend on iteration order. An asynchronous/sequential variant may be a sensitivity check.

Initial parameters (tune; do not present as literature-derived empirical values):

- `epsilon` bounded-confidence threshold: grid 0.05 to 0.50 by 0.05, using the defined total-variation distance.
- `alpha_social`: 0.05 to 0.40 by 0.05.
- `alpha_campaign`: 0.00 to 0.20 by 0.025.
- `alpha_event`: 0.00 to 0.30 by 0.05.
- `w_self`: 0.00 to 0.50 by 0.10.
- Number of simulated voters: use 5,000 for development; check stability at 1,000, 5,000, and 20,000. Scale only if compute permits.
- Campaign and event signal vectors must be valid simplex vectors. The signal strength and timing are assumptions unless supported by data.

Apply clipping/projection only through a tested, documented simplex projection function; do not independently clip probabilities and leave their sum above 1.

### 4.4 Campaign/party agents

Represent each candidate/party as a campaign agent that emits a support vector over candidates at each time step. The baseline campaign process should be intentionally simple and reproducible:

1. Define a campaign calendar in discrete steps (e.g. 30 daily steps if dates are available; otherwise 20 abstract steps and state that they are not calendar days).
2. For each candidate, define a campaign intensity schedule and a target rule. Baseline target rule: all voters receive a common candidate signal. Optional targeted strategy: select voters by a declared observable synthetic segment or graph centrality; do not call a synthetic segment a demographic group.
3. Apply signal strength through `alpha_campaign`; define exact conversion from candidate signal to `campaign_i(t)` and ensure each time step's weights remain valid.
4. Compare no-campaign, uniform-campaign, and targeted-campaign scenarios with the same initial state, graph, and random seed. Treat differences as model scenario comparisons, not estimated real-world causal effects.
5. Record each strategy's parameters, target selection, dates, and seed in experiment metadata.

### 4.5 News/event shocks

Use a scheduler that reads `events.csv` and applies an event vector on its specified time step. Event effects may increase/decrease support for named candidates or issues; ensure the resulting preference is projected onto the simplex. Baseline event scenarios: no event; one mild event affecting a small randomly selected set; one broad event affecting all voters. Report affected fraction, vector, intensity, and duration. Include paired runs with/without the event using identical seed/initial state to estimate model-internal damage spreading. This draws on the shock propagation motivation in reference [3], but the paper's numerical thresholds must not be reused as calibrated election parameters.

### 4.6 Turnout

Use a separate turnout propensity `q_i(t)` in [0,1] rather than treating preference as a guaranteed vote. For baseline, initialize turnout propensity from an explicit synthetic prior (e.g. Beta distribution with mean equal to the official historical turnout rate and concentration parameter 10), and document that this only matches the aggregate rate; it does not recover individual turnout. Optionally vary propensity with campaign mobilization/event signal using a logistic update. At election time, sample turnout as Bernoulli(`q_i(T)`) with a recorded seed. Abstaining agents cast no vote; participating agents vote for the candidate with the largest final support score, sampling categorical preference instead as a sensitivity alternative.

Tune turnout parameters only if there is an independent turnout time series or precinct-level target; otherwise set the aggregate turnout baseline from a training/reference election and report that constraint transparently. For final holdout evaluation, do not tune on the holdout turnout.

## 5. Execution and software architecture

Organize implementation into separable modules (names may differ):

- `config`: validated configuration, parameter defaults, scenario definitions.
- `data`: source metadata, loading, cleaning, and validation.
- `network`: observed/synthetic graph generation and network diagnostics.
- `agents`: voter and candidate/party state.
- `dynamics`: bounded-confidence update, campaign signal, event scheduler, turnout.
- `simulation`: time-step loop, convergence/stop policy, seed control.
- `experiments`: parameter sweeps, replicates, paired scenarios, metadata.
- `metrics`: vote share, turnout, polarization, uncertainty, errors.
- `visualization`: dashboard and static report figures, isolated from core logic.
- `reports`: output tables, run manifests, and reproducibility notes.

Every run must accept a config and seed, write a machine-readable manifest, and save final metrics plus the config. Fix package versions. Use a single centralized random generator with derived, recorded sub-seeds for network, initialization, and turnout. Keep raw data and outputs separate from source code.

## 6. Validation and calibration plan

### 6.1 Model verification (does the code implement the equations?)

Before empirical calibration, verify:

- Every preference vector remains finite, nonnegative, and sums to 1 within tolerance after every update.
- Every turnout probability stays in [0,1].
- With `alpha_social=alpha_campaign=alpha_event=0`, opinions stay fixed.
- With campaign/events disabled and no eligible neighbors, opinions stay fixed.
- With complete confidence (`epsilon` above the maximum possible distance), connected DeGroot dynamics move toward consensus in a strongly connected/appropriately weighted graph.
- With no social links, campaign/event/turnout modules still run and results are deterministic for a fixed seed.
- Same configuration and seed yields the same outputs; changing only the seed changes stochastic runs.
- Vote aggregation, turnout denominators, and candidate mapping reproduce hand-calculated toy examples.

### 6.2 Calibration split and protocol

1. If enough comparable prior elections or independent polling observations exist, define a calibration set and a held-out election/period before tuning. If only one final election exists, calibrate on pre-election polling/precinct variation when available and use the official final constituency result as a limited aggregate holdout; if there is no independent data at all, call the exercise a scenario fit, not out-of-sample validation.
2. Search the declared grid in Section 4, first with a coarse sweep and then refine around the best candidates. Use at least 30 random seeds per setting for final comparisons; use fewer only for exploratory runs and label them.
3. Use common random numbers: for compared configurations, hold graph, initial preferences, and stochastic draws paired by seed where possible.
4. Define a primary objective before looking at the final holdout: mean absolute candidate vote-share error (percentage points) across candidates, with turnout MAE as a separately reported secondary metric. If using polls over time, include poll-share error by date as an additional calibration component with stated weights.
5. Do not optimize polarization to match an outcome unless there is an independent opinion survey measuring polarization. Otherwise report polarization as a simulated outcome and perform sensitivity analysis.
6. Select a stable region of parameter settings, not just the single best run. Report median and 5th-95th percentile over seeds, parameter sensitivity, and whether the ranking changes across network families.

### 6.3 Evaluation metrics

Report all metrics in clearly labeled units:

- **Candidate vote-share MAE (primary):** mean absolute difference in percentage points between simulated and official candidate shares, including Other if used.
- **Candidate vote-share RMSE:** root mean squared percentage-point error; highlights large candidate misses.
- **Winner accuracy:** whether the top candidate matches; meaningful only when there is a valid holdout.
- **Turnout absolute error:** difference in percentage points between simulated and official turnout using the same denominator.
- **Polarization:** compute the average pairwise total-variation distance between sampled voter preference vectors (or an explicitly defined alternative) at initialization and final time; also report candidate-cluster sizes. Describe this as a model diagnostic unless validated by survey data.
- **Opinion change:** average per-voter total-variation distance from initial to final preference.
- **Uncertainty/stability:** metrics' distribution across independent seeds and across reasonable parameter/network variations.
- **Shock propagation (optional):** fraction of agents whose final preference changes by more than 0.01 total-variation distance in paired no-shock/shock runs; report dependence on seed, confidence bound, and whether the perturbed node is high- or low-degree.

A model that matches vote shares only after setting initial preferences to the final vote shares has not demonstrated predictive validity. Compare against simple baselines: (a) uniform/random candidate choice, and (b) persistence of an independent pre-election poll or declared initial preference distribution. Explain baseline data availability.

## 7. Results dashboard and figures

Build a small dashboard for exploration, plus static figures that can be embedded in the report. It should show:

1. Simulation controls: seed, network family, number of agents, confidence threshold, social/campaign/event weights, and scenario.
2. Opinion/support distribution at initial, intermediate, and final time (candidate colors fixed across all charts).
3. Candidate support/vote-share trajectories over time with uncertainty bands across seeds.
4. Network visualization for a small sampled graph, colored by current leading preference; never attempt to draw every edge for a large population.
5. Campaign and event markers on the time series, with a panel showing each applied signal and target fraction.
6. Simulated final result versus official result, vote-share error, turnout, and seed-to-seed interval.
7. A warning/label when the starting opinion distribution or network is synthetic.

Make every plot title/state its units, legend, election/constituency, scenario, and whether values are observed or simulated. Dashboard controls must not silently mutate the saved baseline; provide an explicit run/export action and save config + seed with exported results.

## 8. Eight-week delivery sequence

### Week 1 — Scope, data choice, and model contract

- Confirm election, constituency, candidate handling, unit, data policy, language/toolchain, and repository structure.
- Locate official results and document the source. Check whether precinct, turnout, poll, demographic, and campaign timeline data exist.
- Write `docs/decisions.md`, data dictionary draft, and model equations/update order.
- **Exit gate:** dataset selection and its limitations are explicit; all core variables and candidate representation are defined.

### Week 2 — Data pipeline and baseline graph

- Implement raw-to-processed cleaning/validation and source manifest.
- Implement seeded Watts-Strogatz network baseline and alternative network generators; add diagnostics for degree distribution, components, clustering, and isolates.
- Create synthetic initial preferences with documented options and ranges.
- **Exit gate:** clean data reconciles with official totals; graph/preferences are reproducible for a fixed seed.

### Week 3 — Core opinion simulation

- Implement synchronous bounded-confidence/DeGroot update and convergence logging.
- Add state invariants and toy examples; record per-step support, polarization, and opinion movement.
- **Exit gate:** deterministic baseline runs and all preference-vector invariants hold.

### Week 4 — Campaign, events, and turnout

- Implement candidate campaign schedule, event scheduler, and turnout propensity/sampling.
- Add no-campaign/no-event baselines and paired scenario runs.
- **Exit gate:** every module can be enabled/disabled from config; same-seed paired runs are reproducible.

### Week 5 — Calibration and evaluation

- Freeze the split, primary objective, metrics, parameter ranges, and seed count before tuning.
- Run coarse grid, refine, compare network structures, and summarize seed variability and baselines.
- Record all runs and export a calibration table; avoid tuning on final holdout data.
- **Exit gate:** metrics are computed from a scripted pipeline and validation claim matches the available evidence.

### Week 6 — Dashboard and figures

- Integrate simulation API with dashboard; add timeline, opinion/network views, event/campaign effects, and final result comparison.
- Export report-quality figures from fixed configs and seeds.
- **Exit gate:** clean start-to-finish demo works from documented setup; charts label synthetic assumptions and uncertainty.

### Week 7 — Robustness, report, and stretch decision

- Run sensitivity checks for confidence bound, influence weights, network family, initial preference prior, and turnout assumptions.
- Write methods, data sources, assumptions, results, limitations, and reproducibility sections.
- Decide whether remaining time supports stretch 1. Only start it if core exit gates are met.
- **Exit gate:** a reviewer can reproduce headline figures from saved configs and seeds.

### Week 8 — Final integration, documentation, and demonstration

- Re-run the documented end-to-end workflow from a clean checkout/environment.
- Resolve integration bugs, freeze the final configs and outputs, finalize report/slides/demo script, and tag the version.
- If the core is incomplete, drop all stretch work and finish the single-constituency deliverable.
- **Exit gate:** deliver source, data provenance, configs, reproducible outputs, final report, and a 5-10 minute demo.

## 9. Team of five responsibilities and integration

- **Member A — Network and voter dynamics:** graph generators/diagnostics, preference representation, bounded-confidence update, model specification.
- **Member B — Campaign and party agents:** candidate signals, schedules, targeting scenarios, paired comparisons.
- **Member C — Turnout and events:** turnout propensity, event schema/scheduler, shock experiments.
- **Member D — Data and validation:** official source, cleaning, calibration split/objective, metrics, reproducibility audit.
- **Member E — Integration and communication:** configuration/run interface, dashboard/plots, report, demo, dependency pinning and documentation.

Agree on config/data interfaces in Week 1. Each module owner contributes a small example config and a machine-readable result schema. Integrate continuously from Week 3; do not defer module integration to the final week. Member E coordinates but does not own every integration task alone.

## 10. Stretch goals

Only attempt after the single-constituency acceptance gates are complete.

### Stretch 1 — Multiple constituencies with spillover

- Obtain official results for multiple comparable constituencies and document boundary/time comparability.
- Give each constituency its own voter population and graph; add a cross-constituency influence matrix based on declared geographic adjacency or observed media spillover.
- Tune spillover weight on training constituencies only; compare with independent held-out constituencies.
- Report whether the extension improves held-out accuracy and its added assumptions. Reference [4] provides a hybrid continuous/discrete network opinion mechanism, not a ready-made geographic electoral spillover coefficient.

### Stretch 2 — Multi-phase national simulation

- Build a small synthetic election with explicitly defined phases and phase-to-phase information flow.
- Let early results enter later campaign/event signals only through a declared mechanism and compare with a no-feedback control.
- Use paired runs and shock-spread metrics; label all outputs synthetic. Reference [3] motivates damage-spreading analysis but does not validate national election forecasts.

## 11. Acceptance checklist

The project is complete when:

- [ ] Scope is a working single-constituency simulation with multiple candidate support and a turnout decision.
- [ ] Data source, license/terms, retrieval date, checksum, cleaning, candidate mapping, denominator, and limitations are recorded.
- [ ] Synthetic inputs are clearly distinguished from observations.
- [ ] Network, voter, candidate, campaign, event, turnout, and aggregation behavior are configurable and seeded.
- [ ] Preference and turnout invariants are enforced; deterministic toy examples and fixed-seed behavior are documented.
- [ ] Calibration procedure and holdout decision are written down; baseline comparisons and uncertainty are reported.
- [ ] Vote-share and turnout evaluation use identical definitions/denominators for simulation and official result.
- [ ] Polarization is not presented as empirically validated without independent opinion data.
- [ ] Dashboard and static figures show trajectories, events/campaigns, final comparison, and uncertainty.
- [ ] A clean environment can recreate the processed data, headline runs, and figures from saved commands/configurations.
- [ ] Final report states assumptions, limitations, team contributions, and what would be needed for stronger validation.

## 12. Reference guide

The PDFs below are included under `references/`; use them for mechanism rationale and cite the original publications in the report.

1. **Fortunato et al., “Monte Carlo Simulations of Opinion Dynamics.”** Background on bounded-confidence models, simulation and consensus/polarization phenomena. `references/0501105v1.pdf` (arXiv: [cond-mat/0501105](https://arxiv.org/abs/cond-mat/0501105)). Use for model background, not as evidence that a specific synthetic network describes this electorate.
2. **Charcon and Monteiro, “A multi-agent system to predict the outcome of a two-round election.”** Candidate choice uses public/political factors and neighbor influence; demonstrates one election-modeling approach and real-election reproduction. `references/1-s2.0-S0096300320304409-main.pdf` (Applied Mathematics and Computation 386, 125481, 2020). Its two-round setup is not directly the project's single-constituency specification.
3. **Fortunato, “Damage Spreading and Opinion Dynamics on Scale Free Networks.”** Motivates paired perturbation experiments, network centrality, and measuring how a local shock propagates in bounded-confidence dynamics. `references/0405083v1.pdf` (arXiv: [cond-mat/0405083](https://arxiv.org/abs/cond-mat/0405083)). Treat its thresholds as model-specific, not transferable election parameters.
4. **Xue and Roy, “Averager-copier-voter models for hybrid opinion dynamics in complex networks.”** A hybrid continuous/discrete agent update model and network consensus analysis, potentially useful for an optional richer agent-type or constituency extension. `references/1903.12329v1.pdf` (arXiv: [1903.12329](https://arxiv.org/abs/1903.12329)).
5. **Saburov and Saburov, “Reaching a Nonlinear Consensus: Polynomial Stochastic Operators.”** Formal nonlinear consensus conditions for a structured multi-agent opinion process. `references/s12555-014-0061-0.pdf` (International Journal of Control, Automation, and Systems 12, 1276-1282, 2014; DOI: [10.1007/s12555-014-0061-0](https://doi.org/10.1007/s12555-014-0061-0)). This is theoretical background; it does not specify campaign, turnout, or election data.

## 13. Suggested repository outputs

- `docs/decisions.md` — scope and dated decisions.
- `docs/model.md` — equations, update order, parameter meanings, assumptions.
- `docs/data_dictionary.md` and `data/SOURCES.md` — schema and data provenance.
- `data/raw/` and `data/processed/` — immutable inputs and reproducible cleaned data.
- `configs/` — baseline, calibration, and named scenarios.
- `results/` — run manifests, metrics tables, figures, and calibration summary.
- `reports/` — final report and demo materials.
- `README.md` — environment setup, one-command baseline, calibration, dashboard, and reproduction instructions.
