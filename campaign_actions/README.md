# Campaign actions

Scenario files for candidate campaigns in constituency 1. These are model
controls, not observations. Each file is `{"actions": [ ... ]}`.

| File | What it does |
|---|---|
| `ordinary_c01.json` | CAND_0001 promotes its own platform, days 10-15 |
| `smear_c01.json` | CAND_0001 smears CAND_0003, days 10-15 (same reach/intensity/timing as ordinary, for paired comparison) |
| `multiple_ordinary_c01.json` | CAND_0001 days 5-7 and CAND_0003 days 14-17 |
| `ordinary_uniform_5pct_c01.json` | 5% uniform baseline for matched targeting comparisons |
| `ordinary_students_c01.json` | Ordinary message to 5% of constituency from student voters |
| `ordinary_high_susceptibility_c01.json` | Ordinary message to the 5% most campaign-susceptible voters |
| `ordinary_high_influence_c01.json` | Ordinary message to the 5% with highest modeled network influence |
| `ordinary_community_c01.json` | Ordinary message to one inferred social community |

## Fields

| Field | Meaning |
|---|---|
| `action_id` | Unique name; actions run in `action_id` order |
| `candidate_id` | Sponsoring candidate |
| `start_day`, `duration` | Active on days `start_day` .. `start_day + duration - 1` |
| `intensity` | Signal strength in [0, 1] |
| `reach_fraction` | Share of constituency voters reached, in [0, 1] |
| `message_type` | Action effect: `ordinary` (platform persuasion) or `smear` (fabricated attack in the current example) |
| `target_policy` | `uniform`, `group`, `high_susceptibility`, `high_influence`, or `community` |
| `target_group` | Required for `group`; one of the supported audience labels below |
| `community_id` | Optional for `community`; omit to select a seeded eligible community |
| `target_candidate_id` | Smear only: the candidate being smeared |

Group labels map to voter attributes as follows: `Rural`/`Urban` use
`urban_rural`; `Students`, `Farmers`, and `Unemployed` use `occupation`
(`Farmers` includes Farmer and Agricultural labourer); income labels use
`wealth_level`; `General` matches the whole constituency. Unknown labels are
rejected.

## Message types

- **ordinary**: each active day, recipient opinions move toward the sponsor's
  real platform (mean of its five `position_*` values):
  `x_i += intensity * s_i * lambda_campaign * (platform - x_i)`.
- **smear**: a fabricated negative claim. Opinions do not move. Each active day,
  recipients' utility for the target candidate drops by
  `intensity * s_i * lambda_campaign`. The penalty accumulates and lasts until
  the election. This is the current scenario's truth assumption; the model does
  not fact-check claims or represent truthful opposition research separately.
  Because this penalty does not change opinion or diffuse through the social
  graph, high-influence targeting of a smear currently affects only the reached
  voters' support. It does not model gossip or claim retransmission.

## Recipients

Each action requests `round(N * reach_fraction)` recipients from the
constituency. Recipients are chosen once, before day 1, and receive the action
on every active day. Uniform and group targeting sample without replacement;
the ranking policies select the highest scores and use seeded random
tie-breaking. High susceptibility ranks by the effective susceptibility used
by the campaign update. High influence ranks by weighted out-degree multiplied
by `influence_power`; out-degree counts edges where the voter is the source.
Community labels are inferred once per run using seeded label propagation on
the weak (direction-ignored) projection of the social graph. They are model
clusters, not observed or geographic communities. If `community_id` is omitted,
the action chooses an eligible community with the same seeded action stream.
Explicit IDs refer to the partition generated for that master seed.

Group/community targeting selects only eligible voters. If the requested
budget is larger than the eligible group/community, it selects all eligible
voters and records the shortfall. Run diagnostics include requested/selected counts, coverage,
shortfall, eligible group/community size where relevant, and selected audience
mean susceptibility/influence. They do not publish recipient IDs.

Each action has its own random stream derived from the master seed and its
`action_id`, so adding or reordering actions does not change other actions'
recipients. The 5% targeting examples share an action ID so comparisons use
the same action-specific random stream; ordinary and smear examples also share
an ID to preserve their audience in paired runs.

Because recipients are tied to `action_id`, `ordinary_c01.json` and
`smear_c01.json` deliberately share the same `action_id`: both reach the same
voters, so the only difference between those runs is the campaign effect.
Ranking and random policies select the full requested budget. A group may have
a shortfall if it contains fewer eligible voters than requested.

`lambda_campaign` must be above 0 (see `configs/campaign.json`), otherwise
the run stops with an error.
