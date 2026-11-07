# Synthetic Election Dataset

## 1. Overview and purpose

This project generates a **fictional state** for Multi-Agent Systems (MAS) election simulations. The state contains regions, constituencies, thousands of synthetic voters per constituency, parties and candidates, a directed social network between voters, and political events (campaign, news and external) during an election period.

The generator (`dataset_generator.py`) only produces data. It does **not** implement any election simulation. How the numbers are used is entirely up to the simulation team.

**Everything is synthetic.** No real person, voter, party, candidate, constituency or election is represented. Demographic attributes (age, gender, education, wealth, occupation, urban/rural) are randomly generated and do not describe actual people. All numerical behavioural values are model inputs, not empirical measurements.

## 2. Directory structure

```text
synthetic_election_dataset/
├── voters.csv
├── social_network.csv
├── candidates.csv
└── political_events.csv
```

(The directory name is configurable with `--output-dir` / `OUTPUT_DIR`.)

Project files: `dataset_generator.py`, `README.md`, `USAGE.md`, `requirements.txt`.

## 3. IDs and geography

* `region_id`: integers `1 … NUM_REGIONS`.
* `constituency_id`: integers `1 … total constituencies`, numbered consecutively; constituencies are assigned to regions in contiguous blocks. If the total is not divisible by the number of regions, earlier regions get one extra constituency (e.g. 8 constituencies in 3 regions → 3, 3, 2).
* Neighbouring constituencies are those whose ids differ by 1 (including across a region boundary). This is the only place cross-constituency social links may occur.
* `voter_id`: `C<constituency>_V<index>`, zero-padded, e.g. `C05_V0001`. Padding grows with the configuration (3000 voters → 4 digits; 100 voters → 3 digits).
* `candidate_id`: `CAND_0001`, … unique across the whole state.
* `edge_id`: `E0000001`, …. `event_id`: `EVT_00001`, … (events are numbered in chronological order).
* `party_id`: `P1 … Pn` for the statewide parties (same identity in every constituency); `IND` for independent candidates.

Every record in voters, social network and candidates carries `constituency_id` and `region_id`. Events carry them only when their scope requires it (see below).

## 4. Files and columns

Float values are written with 4 decimals. Missing values are empty cells (pandas reads them as `NaN`).

### 4.1 voters.csv — one row per voter agent

| Column | Type | Values / range | Meaning |
|---|---|---|---|
| `voter_id` | string | unique | Voter identifier |
| `constituency_id` | int | 1…C | Constituency of residence |
| `region_id` | int | 1…R | Region of that constituency |
| `age_group` | category | `18-24`, `25-34`, `35-44`, `45-59`, `60+` | |
| `gender` | category | `Male`, `Female`, `Other` | |
| `education_level` | category (ordinal) | `No formal`, `Primary`, `Secondary`, `Graduate`, `Postgraduate` | |
| `wealth_level` | category (ordinal) | `Low`, `Middle`, `High` | |
| `occupation` | category | Farmer, Agricultural labourer, Unskilled labourer, Skilled worker, Private employee, Government employee, Business owner, Professional, Student, Homemaker, Unemployed, Retired | |
| `urban_rural` | category | `Urban`, `Rural` | |
| `initial_opinion` | float | [-1, 1] | Starting political leaning on a one-dimensional axis. Parties also have positions on this axis (see §5); 0 is the centre. |
| `event_sensitivity` | float | [0, 1] | How strongly this voter reacts to political events |
| `turnout_propensity` | float | [0, 1] | Baseline tendency to actually vote |
| `susceptibility` | float | [0, 1] | How responsive the voter is to influence from connected voters |
| `influence_power` | float | (0, `INFLUENCE_POWER_MAX`], default max 5.0 | Ability of the voter to influence connected voters |
| `confidence_threshold` | float | [0, 2] | Width of the opinion gap within which the voter is willing to be influenced |
| `issue_employment`, `issue_economy`, `issue_education`, `issue_healthcare`, `issue_agriculture` | float | [0, 1] | Importance the voter attaches to each issue |

### 4.2 social_network.csv — one row per directed connection

| Column | Type | Meaning |
|---|---|---|
| `edge_id` | string | Unique edge id |
| `constituency_id` | int | Constituency of the **target** voter |
| `region_id` | int | Region of the **target** voter |
| `source_voter_id` | string | Voter who **influences** the other |
| `target_voter_id` | string | Voter who **receives** influence |
| `edge_weight` | float, (0, 1] | Strength of the tie; depends (probabilistically) on relationship type |
| `relationship_type` | category | `Family`, `Friends`, `Colleagues`, `Community`, `Other` |

**Direction:** an edge `A → B` (`source_voter_id = A`, `target_voter_id = B`) means *A influences B*. A reciprocal tie exists only if the reverse row `B → A` also exists. Retrieve who influences a voter by filtering on `target_voter_id`; whom a voter influences by filtering on `source_voter_id`.

**Structure:**

* Each constituency is split into several communities of unequal size (target size `AVG_COMMUNITY_SIZE`).
* For each edge, the source is in the target's own community by default; with probability `CROSS_COMMUNITY_CONNECTION_PROBABILITY` it is in another community of the same constituency; with probability `CROSS_CONSTITUENCY_CONNECTION_PROBABILITY` it is in a neighbouring constituency (id ± 1). Cross-constituency edges are identified by comparing the source voter's constituency (from the `voter_id` / voters.csv) with the edge's `constituency_id`.
* Each voter's **in-degree** is drawn from a skewed distribution (mean ≈ `AVG_CONNECTIONS_PER_VOTER`, at least 1), so total edges ≈ voters × `AVG_CONNECTIONS_PER_VOTER`. Sources are chosen proportionally to `influence_power ** INFLUENCE_DEGREE_EXPONENT`, so **out-degree varies strongly** and influential voters have many outgoing edges; some voters have no outgoing edges.
* The network is sparse. There are no self-edges and no duplicate directed edges.

`influence_power` and `susceptibility` live in voters.csv, not on the edges.

### 4.3 candidates.csv — one row per candidate

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | string | Unique across the state |
| `constituency_id`, `region_id` | int | Where the candidate contests |
| `party_id`, `party_name` | string | `P1…Pn` or `IND` / `Independent` |
| `is_incumbent` | bool (`True`/`False`) | At most one incumbent per constituency (probability `INCUMBENT_PROBABILITY`) |
| `experience_level` | int 1–5 | 1 = first-time candidate … 5 = veteran; incumbents are biased upwards |
| `local_popularity` | float [0, 1] | Local standing, driven by how well the candidate's party fits local voters' opinions, plus incumbency and experience |
| `position_employment`, `position_economy`, `position_education`, `position_healthcare`, `position_agriculture` | float [-1, 1] | Policy stance per issue on the same [-1, 1] axis as `initial_opinion` |

Each constituency has exactly `NUM_CANDIDATES_PER_CONSTITUENCY` candidates. If this is ≤ the number of parties, a subset of parties contests (locally stronger parties are more likely to appear); if it is larger, all parties contest and the remaining slots are independents. A party never fields two candidates in one constituency. Candidates of the same party have similar (not identical) positions around the party's statewide positions; independents have scattered positions.

### 4.4 political_events.csv — one row per event

The generator creates a fixed number of schedule rows, rather than sampling an
event-arrival process: 12 local rows for each of 21 constituencies (252), 6
rows for each of 3 regions (18), and 8 state-wide rows, for 278 total. A
constituency is loaded with its 12 local rows, its region's 6 rows, and all 8
state rows (26 total). Rows are records at different geographic scopes, so
278 is not the number of events seen by one constituency. These defaults make
a fairly busy synthetic schedule; they are scenario settings, not an estimate
of a real election's event rate. The counts can be changed with the generator
options documented in `USAGE.md`.

| Column | Type | Meaning |
|---|---|---|
| `event_id` | string | Unique, chronological |
| `region_id` | int or empty | Empty for state-level events |
| `constituency_id` | int or empty | Empty for state- and region-level events |
| `event_day` | int | 1 … `ELECTION_DURATION_DAYS` |
| `source_type` | category | `Candidate`, `News`, `External` |
| `event_type` | category | Manifesto announcement, Campaign rally, Policy announcement, Controversy, Economic development, Employment announcement, Agricultural issue, Healthcare issue, Education initiative, External economic shock, Other |
| `issue` | category | `employment`, `economy`, `education`, `healthcare`, `agriculture`, or `general` |
| `candidate_id` | string or empty | Optional associated candidate |
| `intensity` | float [0, 1] | Strength/prominence of the event |
| `credibility` | float [0, 1] | How believable the information is (External > News > Candidate on average) |
| `duration` | int ≥ 1 | Active days, starting on `event_day`; event effects are apportioned over these days |
| `target_group` | category | `General`, `Rural`, `Urban`, `Students`, `Farmers`, `Unemployed`, `Low-income`, `Middle-income`, `High-income` |
| `candidate_effect` | float [-1, 1] or empty | Signed effect on the **associated candidate** (empty when `candidate_id` is empty) |
| `description` | string | Human-readable text |

Optional event-behavior columns can be supplied without changing the base
generator CSV: `memory_half_life_days` (nonnegative; 0 means persistent),
`reminder_probability` (0–1 chance per post-duration day),
`occurrence_probability` (0–1 chance the scheduled event happens),
`depends_on_event_id` and `condition_probability` (chance of this event if its
earlier dependency happened), and `structural_effect` (`none` or
`candidate_withdrawal`). Candidate-free shocks may also set `shock_mode`
(`balanced` or `directional`) and `opinion_direction` in [-1, 1]; positive
values shift opinion toward the positive end of the model axis and negative
values toward the negative end. When omitted, the simulation config supplies the
defaults: 7-day memory half-life, no reminders, and certain occurrence. A
withdrawal is permanent for the election and removes the candidate from
support/vote consideration after its start day.

An example moderate C01 schedule uses 13 relevant events (6 local, 3 regional,
4 state-wide) and demonstrates these optional fields in
`event_scenarios/c01_moderate.csv`. It does not modify the checked-in generated
278-row schedule.

**Scopes**

| Scope | `region_id` | `constituency_id` | Possible sources | Count |
|---|---|---|---|---|
| Constituency | set | set | Candidate, News, External | `EVENTS_PER_CONSTITUENCY` per constituency |
| Region | set | empty | Candidate, News, External | `EVENTS_PER_REGION` per region |
| State | empty | empty | News, External | `STATE_LEVEL_EVENTS` in total |

Candidate-sourced events always name a candidate (from that constituency or region). News events sometimes concern a candidate. External events and all state-level events name no candidate and carry no `candidate_effect`.

**Distributions:** intensity, credibility and effect are drawn from Beta/Normal distributions whose means depend on event type and source, so two events of the same type differ. `candidate_effect` can be positive, negative or near zero (e.g. controversies skew negative, rallies skew positive, but with large spread). It is a signal about the *candidate*, not a claim that the event is good or bad for every voter; interpretation is left to the simulation team. Campaign events cluster later in the election period; news/external events are spread evenly. The `issue` of an event is weighted toward what local voters care about.

In the current simulator, events are activated on each scheduled active day. The
total signal is split evenly across the event's duration. Candidate-linked
events add a signed utility adjustment to that candidate; candidate-free
events apply a seeded, zero-mean opinion direction to their audience. After the
active duration, those effects decay by their configured half-life, and an
optional per-day reminder can reinforce them. `lambda_event` controls signal
strength separately from social and campaign strength. Paired shock/no-shock
runs use identical random streams. The event text is descriptive metadata; the
simulator does not use an LLM or infer effects from that text.

## 5. Behavioural features — how they differ

* **`initial_opinion`**: where the voter starts on the [-1, 1] opinion axis. Parties sit at fixed positions on the same axis (P1 around -0.7 … last party around +0.7 with the default three parties), but no party preference column is stored; the team decides how opinion maps to party choice.
* **`influence_power`**: how much *this voter affects others*. Heavy-tailed; higher for educated, wealthier or higher-status occupations. Also determines how many outgoing edges the voter tends to have.
* **`susceptibility`**: how much *this voter is moved by others*. Independent of influence power (generated from its own distribution; mildly related to education and age).
* **`event_sensitivity`**: how much *this voter responds to political events* (not to other voters).
* **`turnout_propensity`**: baseline probability-like tendency to *cast a vote* at all; increases with age and education, and varies by constituency.
* **`confidence_threshold`**: the opinion distance within which the voter considers others' views (bounded-confidence style). Larger = more open; voters with extreme opinions tend to have smaller thresholds.

## 6. Which attributes are generated, and how

All attributes are generated. Generation is hierarchical:

1. **State**: baseline urbanisation, education, wealth and issue importance.
2. **Region**: deviations from the state (urbanisation, education, wealth, agricultural intensity, issue salience, political lean, turnout, region-specific effects of demographics on opinion).
3. **Constituency**: deviations from the region, plus a political *archetype*: `strong` (one party dominates), `moderate` (mild preference), `competitive` (broad, centred distribution) or `polarized` (two opinion clusters). Constituencies in one region are similar but not identical.
4. **Voter**: demographics are sampled from the constituency profile with probabilistic dependencies (urban → younger and more educated; education → wealth; occupation depends on urbanity, education, wealth, age and gender; issue importance depends on occupation, age, wealth). Opinions are sampled from the constituency's opinion mixture, then shifted by demographics. Behavioural parameters use Beta/log-normal distributions around demographic-dependent means. Individual noise keeps within-constituency variation large.

Network communities, candidates (popularity from local opinions) and events (issues from local salience) are then generated from these profiles.

## 7. Reproducibility and validation

The same configuration and seed produce byte-identical CSVs; a different seed yields a different election with the same structural rules. Randomness comes from NumPy `default_rng` streams spawned from one `SeedSequence`.

Before writing, the generator validates: unique IDs; valid constituency/region references and mapping; value ranges and categories; no missing mandatory values; edge endpoints exist, no self or duplicate edges, positive weights, cross-constituency edges only between neighbours; candidate count per constituency, party IDs, one incumbent at most; event days, candidate references, scope consistency. A violation raises `ValidationError` and no files are written.

## 8. Assumptions and limitations

* Opinion is one-dimensional; party positions and candidate positions share that axis. This is a simplification.
* The number of voters is identical in every constituency.
* Neighbouring constituencies are defined by consecutive ids, not real geography.
* Population shares, correlations and effect sizes are plausible-looking assumptions chosen for variety, not calibrated to any real data. They are defined as constants at the top of `dataset_generator.py`.
* The realised number of edges differs a little from `voters × AVG_CONNECTIONS_PER_VOTER` because degrees are random.
* Edge `constituency_id`/`region_id` refer to the target voter.
* `candidate_effect` is generated independently of voter attributes; it is only a signal.
* Do not use the data to draw conclusions about real elections or real people.
