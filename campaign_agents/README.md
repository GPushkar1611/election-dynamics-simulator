# Candidate agents

Agent files make candidates **choose** their campaign during the run, instead of following a fixed script (`campaign_actions/`). The code is in `election_sim/agents/`.

Each file is `{"agents": [ ... ]}` with at most one agent per candidate. Candidates without an agent do not campaign, unless you also pass scripted actions.

| File | What it does |
|---|---|
| `c01_agents.json` | C01 from day 10: CAND_0001 (leader) `rebut`, CAND_0002 `greedy` with misleading messages, CAND_0003 (trailing) `go_negative` with high-influence targeting |

Run it (from the repository root):

```powershell
python -m election_sim --config configs/agents.json --agents campaign_agents/c01_agents.json --seed 42 --output results/agents_seed42
```

Add `--overwrite` to run again into the same folder.

`configs/agents.json` is `configs/campaign.json` plus `smear_spread_probability`, `poll_sample_size` and `poll_noise`. `lambda_campaign` must be above 0.

## How an agent works

On each decision day (`start_day`, then every `decide_every` days):

1. **Observe.** Before anything else happens that day, the simulation polls `poll_sample_size` random voters on yesterday's state. Each respondent names their highest-support candidate, and the shares get Gaussian noise (`poll_noise`). The poll also gives shares per targeting group (Rural, Students, Farmers, ...; groups with fewer than 10 respondents are left out). All candidate agents deciding that day receive this same poll; voters do not see it unless a public poll event is released.
2. **Decide.** The agent's strategy turns the poll into one `CampaignAction` lasting `decide_every` days (cut at election day). Its ID is `<candidate>_dayDD_<message_type>`.
3. **Act.** Recipients are selected with the normal targeting code. The cost is recipients x active days. If the agent cannot afford it from `budget`, the action is skipped (logged with `taken: false`).

## Strategies

| Strategy | Decision |
|---|---|
| `greedy` | Message to the group where the candidate polls worst (`target_policy: group`). Uniform if the poll has no group data. |
| `go_negative` | If trailing the poll leader by more than `smear_gap`: smear the leader, targeted with `policy`. Otherwise like `greedy`. |
| `rebut` | Historical name. If own poll share fell by more than `drop_alert` since its previous poll, send a broad message using `policy`. It does not identify or verify the cause of the drop. Otherwise like `greedy`. |

With `distortion` > 0 the agent's own messages are `misleading` instead of `ordinary` (see `campaign_actions/README.md`).

## Fields

| Field | Default | Meaning |
|---|---|---|
| `candidate_id` | required | Candidate the agent campaigns for |
| `strategy` | `greedy` | `greedy`, `go_negative` or `rebut` |
| `budget` | 3000 | Total voter contacts (recipients x active days) |
| `start_day` | 1 | First decision day |
| `decide_every` | 3 | Days between decisions; also each action's duration |
| `reach_fraction` | 0.05 | Share of the constituency each action asks to reach |
| `intensity` | 0.5 | Message strength in [0, 1] |
| `distortion` | 0 | > 0 makes own messages misleading |
| `policy` | `uniform` | Targeting for smears and rebuttals: `uniform`, `high_susceptibility`, `high_influence`, `community` |
| `smear_gap` | 0.10 | `go_negative`: poll gap that triggers a smear |
| `drop_alert` | 0.02 | `rebut`: poll drop that triggers a rebuttal |

## What gets recorded

- `manifest.json` -> `diagnostics.agent_decisions`: one row per decision (day, strategy, message, target, poll share, recipients, cost, taken, budget left).
- `diagnostics.polls`: every poll the agents saw.
- `diagnostics.agent_budget_left` and `metadata.agents`: the agent settings.
- Agent actions also appear in `diagnostics.campaign_audiences` and, with `--record-history`, in `history.npz` and `history_timeline.json` like scripted actions.

## Notes

- The ranking policies (`high_influence`, `high_susceptibility`) rank fixed voter attributes, so repeated actions reach the same top voters every time.
- Agents decide in candidate ID order, and all deciding agents see the same poll on a given day.
- Agents use their own poll random stream. A run with agents is reproducible from its seed, and a run without agents is identical to one made before agents existed.
