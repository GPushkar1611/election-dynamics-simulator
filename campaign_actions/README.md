# Campaign actions

Scenario files for candidate campaigns in constituency 1. These are model
controls, not observations. Each file is `{"actions": [ ... ]}`.

| File | What it does |
|---|---|
| `ordinary_c01.json` | CAND_0001 promotes its own platform, days 10-15 |
| `smear_c01.json` | CAND_0001 smears CAND_0003, days 10-15 (same reach/intensity/timing as ordinary, for paired comparison) |
| `multiple_ordinary_c01.json` | CAND_0001 days 5-7 and CAND_0003 days 14-17 |

## Fields

| Field | Meaning |
|---|---|
| `action_id` | Unique name; actions run in `action_id` order |
| `candidate_id` | Sponsoring candidate |
| `start_day`, `duration` | Active on days `start_day` .. `start_day + duration - 1` |
| `intensity` | Signal strength in [0, 1] |
| `reach_fraction` | Share of constituency voters reached, in [0, 1] |
| `message_type` | Truth label: `ordinary` (truthful) or `smear` (fabricated) |
| `target_policy` | Who is reached; only `uniform` for now |
| `target_candidate_id` | Smear only: the candidate being smeared |

## Message types

- **ordinary**: each active day, recipient opinions move toward the sponsor's
  real platform (mean of its five `position_*` values):
  `x_i += intensity * s_i * lambda_campaign * (platform - x_i)`.
- **smear**: a fabricated negative claim. Opinions do not move. Each active day,
  recipients' utility for the target candidate drops by
  `intensity * s_i * lambda_campaign`. The penalty accumulates and lasts until
  the election.

## Recipients

Recipients are chosen once, before day 1, and receive the action on every
active day. Each action has its own random stream derived from the master
seed and its `action_id`, so adding or reordering actions does not change
other actions' recipients.

Because recipients are tied to `action_id`, `ordinary_c01.json` and
`smear_c01.json` deliberately share the same `action_id`: both reach the same
voters, so the only difference between the two runs is the message type. Every targeting policy picks the same number of
voters, `round(N * reach_fraction)`.

`lambda_campaign` must be above 0 (see `configs/campaign.json`), otherwise
the run stops with an error.
