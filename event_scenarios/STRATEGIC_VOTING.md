# Strategic voting after public polls (epistemic logic)

Code: `election_sim/strategic.py`, connected in `election_sim/simulation.py`.
Tests: `tests/test_strategic.py`. Off by default (`strategic_voting = 0`).

## Idea

What voters *know* decides whether they vote strategically:

| Poll | Who knows the result | Epistemic status | Effect on voters |
|---|---|---|---|
| Candidate decision-day poll (`diagnostics.polls`) | All candidate agents deciding that day; not voters | Shared agent information | None |
| A `poll_release` news event (`diagnostics.public_polls`) | Every voter | Public announcement, interpreted as common knowledge | Strategic voting |

The model treats a public poll as common knowledge, then applies a simplified utility penalty. It does not represent individual knowledge states or possible worlds, so this is an epistemic-logic-inspired mechanism rather than a formal epistemic model.

## Rule

When a poll is published:

1. **Viable** candidates are the top two in the published poll. Ties go to the earlier candidate ID.
2. Every voter who received the announcement lowers their utility for each non-viable candidate by `strategic_voting`. Support (the softmax over utilities) is recomputed the same day.
3. A later public poll replaces this knowledge; it does not add to it.

Notes:

- Voters who like a non-viable candidate much more than the rest still vote for it. A larger `strategic_voting` means more deserters.
- Voters react to the **published numbers, noise included**, not to the true state. A noisy poll can mislabel a candidate as hopeless.
- With two candidates, both are always viable, so nothing changes.
- `lambda_event = 0` disables all news, including poll releases, and so also strategic voting.
- It works the same in regional runs: each constituency reacts to its own poll releases.

## Run it

`configs/strategic.json` is `configs/news.json` with `strategic_voting: 1.0`. The schedule `c01_news.csv` publishes a poll on day 25.

```powershell
python -m election_sim --config configs/strategic.json --campaign-actions campaign_actions/smear_c01.json --events-file event_scenarios/c01_news.csv --seed 42 --output results/strategic_seed42
python -m election_sim --config configs/news.json --campaign-actions campaign_actions/smear_c01.json --events-file event_scenarios/c01_news.csv --seed 42 --output results/sincere_seed42
```

Add `--overwrite` to run again into the same folder.

## How to tell it works

Seed 42, C01. The day-25 poll shows CAND_0002 at about 2%, so it is non-viable.

| `strategic_voting` | CAND_0001 | CAND_0002 | CAND_0003 |
|---|---|---|---|
| 0 (sincere, `news.json`) | 1048 | 122 | 379 |
| 1.0 (`strategic.json`) | 1098 | 58 | 393 |

Where to look:

- `manifest.json` → `diagnostics.public_polls`: the published shares, plus `non_viable` and `switched_top_choice` (voters whose single favourite candidate changed).
- `daily_support.csv`: CAND_0002's mean support drops from 0.070 on day 24 to 0.027 on day 25. Before day 25 both runs are identical.

In C01, `switched_top_choice` is 0: no voter has CAND_0002 as their favourite. Its votes come from voters whose support is split, because a vote is drawn from the voter's support for every candidate. Strategic voting removes that partial support. In a closer three-way race, `switched_top_choice` is positive; the tests show this.

## Scope

The current code does not implement Kripke models or individual private polls.
It shares decision-day poll data among candidate agents, keeps it hidden from
voters, and treats a `poll_release` as a public announcement. A formal
possible-worlds example can be included as course context, but should be clearly
labelled as theory rather than implemented behavior. Compare
`strategic_voting=0` and `>0` using matched seeds to show the implemented rule.
