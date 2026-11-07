# Campaign actions

Scripted scenario files for candidate campaigns in constituency 1: every
action is fixed in advance. These are model controls, not observations. Each
file is `{"actions": [ ... ]}`. For candidates that choose their own actions
during the run from polls, see `campaign_agents/README.md`.

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
| `message_type` | Action effect: `ordinary` (platform persuasion), `misleading` (distorted platform) or `smear` (fabricated attack) |
| `distortion` | Misleading only, in [0, 1]: how far the claimed platform moves toward the audience (default 0) |
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
- **misleading**: like ordinary, but the message misrepresents the platform.
  It claims a position moved toward the audience's own mean opinion:
  `q = platform + distortion * (audience_mean - platform)`, then
  `x_i += intensity * s_i * lambda_campaign * (q - x_i)`. `distortion = 0` is
  exactly an ordinary message.
- **smear**: a fabricated negative claim ("candidate X did something bad").
  Opinions do not move. Recipients come to **believe** the claim, and each
  active day their smear strength for the target candidate grows by
  `intensity * s_i * lambda_campaign`. While a voter believes the claim, that
  strength is subtracted from their utility for the target. It lasts until the
  election. The model does not fact-check claims.
- **Smear spreading (word of mouth)**: off by default. With
  `smear_spread_probability = p > 0` in the config, every believer passes the
  claim along each of their outgoing ties each day with probability
  `p * edge_weight`. A voter who had not heard it believes it with probability
  equal to their susceptibility; otherwise they reject it for good. A new
  believer gets the teller's strength times the tie weight, so rumours weaken
  as they travel. This makes high-influence targeting of smears matter. Each
  voter's state per claim (0 unaware, 1 believes, 2 rejects) is in
  `final_state.csv` (`smear_belief_<candidate>`), the daily
  `smear_believers` count is in `daily_summary.csv`, and the full daily state
  is in `history.npz` (`smear_belief`).

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
