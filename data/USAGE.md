# Usage Guide

For the simulation team. See `README.md` for the meaning of every column.

All values in the CSV files are **synthetic parameters** to be treated as model inputs, not empirically measured facts. This guide does not prescribe any simulation algorithm.

## 1. Prerequisites

* Python 3.12.10 and the pinned packages in the repository root `requirements.txt` for byte-identical dataset reproduction
* `numpy` and `pandas`

## 2. Install

```bash
pip install -r requirements.txt
```

## 3. Generate the dataset

```bash
python dataset_generator.py
```

This writes `synthetic_election_dataset/{voters,social_network,candidates,political_events}.csv` (default: 3 regions, 21 constituencies, 63,000 voters, ~500k edges; takes a few seconds) and prints a summary. If validation fails, an error is printed, the exit code is 1 and nothing is written.

## 4. Configure

Two ways, no changes to the generation logic are needed:

**A. Edit the constants** in the `CONFIGURATION` section at the top of `dataset_generator.py` (`NUM_REGIONS`, `VOTERS_PER_CONSTITUENCY`, …).

**B. Command-line overrides** (these win over the constants):

| What | Flag | Constant |
|---|---|---|
| Regions | `--num-regions` | `NUM_REGIONS` |
| Constituencies per region | `--constituencies-per-region` | `CONSTITUENCIES_PER_REGION` |
| Total constituencies (any number ≥ regions; split as evenly as possible) | `--num-constituencies` | `NUM_CONSTITUENCIES` |
| Voters per constituency | `--voters-per-constituency` | `VOTERS_PER_CONSTITUENCY` |
| Candidates per constituency | `--candidates-per-constituency` | `NUM_CANDIDATES_PER_CONSTITUENCY` |
| Statewide parties | `--num-parties` | `NUM_POLITICAL_PARTIES` |
| Incumbent probability | `--incumbent-probability` | `INCUMBENT_PROBABILITY` |
| Election length (days) | `--election-days` | `ELECTION_DURATION_DAYS` |
| Constituency-level events (each) | `--events-per-constituency` | `EVENTS_PER_CONSTITUENCY` |
| Region-level events (each region) | `--events-per-region` | `EVENTS_PER_REGION` |
| State-level events (total) | `--state-events` | `STATE_LEVEL_EVENTS` |
| Average connections per voter | `--avg-connections` | `AVG_CONNECTIONS_PER_VOTER` |
| Target community size | `--avg-community-size` | `AVG_COMMUNITY_SIZE` |
| Cross-community probability | `--cross-community-prob` | `CROSS_COMMUNITY_CONNECTION_PROBABILITY` |
| Cross-constituency probability | `--cross-constituency-prob` | `CROSS_CONSTITUENCY_CONNECTION_PROBABILITY` |
| Upper bound of `influence_power` | `--influence-power-max` | `INFLUENCE_POWER_MAX` |
| Random seed | `--seed` | `RANDOM_SEED` |
| Output directory | `--output-dir` | `OUTPUT_DIR` |

Totals (voters, candidates, expected edges, events) are computed automatically; the summary prints the actual edge count next to the expected one.

Examples:

```bash
# Larger population and different seed
python dataset_generator.py --voters-per-constituency 5000 --seed 123

# 4 regions, 10 constituencies (uneven split: 3, 3, 2, 2), 5 candidates, 12 days
python dataset_generator.py --num-regions 4 --num-constituencies 10 \
    --candidates-per-constituency 5 --election-days 12

# Small test dataset
python dataset_generator.py --num-regions 2 --num-constituencies 4 \
    --voters-per-constituency 100 --avg-community-size 25 --output-dir tiny_dataset

# Denser network, more events
python dataset_generator.py --avg-connections 15 --events-per-constituency 25
```

**Seeds:** the same configuration and seed reproduce byte-identical files; another seed gives a different election with the same structure. Always record the configuration and seed with your results.

Notes: with `--candidates-per-constituency` larger than `--num-parties`, extra candidates are independents (`party_id = IND`). If the voter count per constituency is small, constituencies may contain only one community (no cross-community edges); lower `--avg-community-size` to get several.

## 5. Load the files

```python
import pandas as pd

d = "synthetic_election_dataset/"
voters = pd.read_csv(d + "voters.csv")
social_network = pd.read_csv(d + "social_network.csv")
candidates = pd.read_csv(d + "candidates.csv")
events = pd.read_csv(d + "political_events.csv",
                     dtype={"region_id": "Int64", "constituency_id": "Int64"})  # keeps IDs as ints despite empties

print(voters.shape, social_network.shape, candidates.shape, events.shape)
voters.info()
print(voters.describe())
```

## 6. Query examples

Voter IDs look like `C05_V0001` (zero-padding depends on the configuration; check `voters["voter_id"].iloc[0]`).

```python
# Voters of constituency 5 / region 2
constituency_voters = voters[voters["constituency_id"] == 5]
region_voters = voters[voters["region_id"] == 2]

# A voter's connections
vid = constituency_voters["voter_id"].iloc[0]
influences_others = social_network[social_network["source_voter_id"] == vid]   # vid -> others
influenced_by = social_network[social_network["target_voter_id"] == vid]       # others -> vid
all_ties = social_network[(social_network["source_voter_id"] == vid) |
                          (social_network["target_voter_id"] == vid)]

# Candidates contesting in constituency 5
cands = candidates[candidates["constituency_id"] == 5]

# Events affecting constituency 5: its own, its region's, and state-level
region_of_5 = int(voters.loc[voters["constituency_id"] == 5, "region_id"].iloc[0])
affecting_5 = events[
    (events["constituency_id"] == 5) |
    (events["constituency_id"].isna() & (events["region_id"] == region_of_5)) |
    (events["constituency_id"].isna() & events["region_id"].isna())
]

# Events for a region (constituency- and region-level events), or state-level only
region_events = events[events["region_id"] == 2]
state_events = events[events["region_id"].isna() & events["constituency_id"].isna()]

# Cross-constituency edges (source voter lives in a different constituency than the edge's)
src_cons = social_network["source_voter_id"].map(voters.set_index("voter_id")["constituency_id"])
cross_edges = social_network[src_cons != social_network["constituency_id"]]
```

## 7. Using the files to initialise a simulation

How the values are interpreted and updated is up to you. The files map onto typical simulation components as follows:

* **Agents** – one agent per row of `voters.csv`. Demographics, `initial_opinion`, issue importances and the behavioural parameters (`susceptibility`, `influence_power`, `confidence_threshold`, `event_sensitivity`, `turnout_propensity`) are agent attributes. Note the distinction: `influence_power` concerns a voter's effect on others, `susceptibility` concerns the voter's responsiveness to others, `event_sensitivity` the response to events.
* **Network** – `social_network.csv` is a directed weighted edge list (`source → target`, source influences target). Example with NetworkX (optional dependency):

  ```python
  import networkx as nx
  G = nx.from_pandas_edgelist(social_network, "source_voter_id", "target_voter_id",
                              edge_attr=["edge_weight", "relationship_type"],
                              create_using=nx.DiGraph)
  G.add_nodes_from(voters["voter_id"])          # include voters with no edges
  ```

  A pure-pandas alternative is to group edges by `target_voter_id` to get each agent's list of influencers and their `edge_weight`s.
* **Candidates / parties** – `candidates.csv` provides per-constituency candidate profiles: party, incumbency, experience, popularity and policy positions (same [-1, 1] axis as `initial_opinion`; compare with voter issue importances or opinions however your model requires).
* **Event system** – `political_events.csv` is an event schedule: `event_day` (start), `duration`, geographic scope (via `region_id`/`constituency_id`), `target_group`, `issue`, `intensity`, `credibility`, and optionally a candidate and a signed `candidate_effect`. Mapping `target_group` values (e.g. `Farmers`, `Low-income`) to voters via their attributes is a modelling choice for your team.

## 8. Troubleshooting

* `ValidationError`/"Dataset validation failed": a consistency check failed; the message names the rule. Please report it together with the configuration and seed.
* "Invalid configuration": a parameter is impossible (e.g. fewer constituencies than regions, probabilities summing above 1).
* Network edge count differs from the expected figure by a few percent: degrees are random; this is normal.
