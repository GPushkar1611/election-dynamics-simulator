# Synthetic issue preference state

This note explains the optional issue-profile sensitivity mode. The default
model remains scalar.

## Purpose

Candidate support can compare a voter with a candidate separately on the five
available issues (employment, economy, education, healthcare, agriculture).
The supplied voter data records each issue's importance, but not the voter's
position on that issue. The simulator therefore offers an explicitly
synthetic, opt-in preference state for sensitivity analysis. It does not
replace or add columns to the supplied voter CSVs.

## State and initialization

With `support_mode: "synthetic_issue_profile"`, initialization creates a
five-value preference vector for each voter:

`issue_preference[i, k] = clip(initial_opinion[i] + issue_offset[i, k], -1, 1)`

The offsets are generated reproducibly from a dedicated random stream. They
average to zero across the five issues before clipping, and their maximum
absolute magnitude is controlled by `synthetic_issue_dispersion` in [0, 1].
The offset pattern is fixed within a run. Dispersion zero means no individual
issue offsets, but scoring is still issue-aware and is not equivalent to the
scalar support formula.

## Daily updates

The scalar opinion remains the state used by bounded-confidence social
influence and ordinary campaign effects. At the end of each simulated day,
the change in scalar opinion is added to all five issue preferences, with
each value clipped to [-1, 1]. This moves the broad center of the voter's
profile while preserving its issue-specific pattern, except where clipping
applies. Candidate support is then recalculated using the current five-value
profile and that voter's normalized `issue_*` importance weights.

The issue-aware candidate distance is the weighted sum of absolute distances
between voter preference and candidate position over the five issues. The
existing scalar support mode remains the default and is unchanged.

## Output and interpretation

Issue-aware run `final_state.csv` includes `synthetic_issue_offset_<issue>`
and `final_issue_preference_<issue>` columns. The manifest records the support
mode and marks the generated beliefs as synthetic. These coordinates are not
observations, and issue-position signs do not yet have named political
meanings. They should be used to examine sensitivity to assumptions, not to
claim that the modeled profiles describe real voters.

## Event behavior

In issue-aware mode, a neutral event whose `issue` is one of the five named
issues updates only that preference dimension. It uses that event's existing
audience, issue importance, event sensitivity, intensity, credibility,
duration, memory, and reminder settings. Directional events should provide
`issue_direction` in [-1,1]. Older schedules may use `opinion_direction` as a
compatibility fallback for an issue-specific event. In scalar mode, the same
issue-specific event continues to shift scalar opinion along the old path.

A neutral event with `issue=general` still moves scalar opinion; in issue-aware
mode that moves the broad center of all five preferences. Candidate-associated
events continue to adjust that candidate's utility directly, including when
they name a specific issue. This is a declared simulation rule, not a
validated mapping from named news to real political behavior.
