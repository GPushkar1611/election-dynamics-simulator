# Initial campaign actions

`ordinary_c01.json` is the first campaign scenario for constituency 1.

An `ordinary` action uses the candidate's actual platform position, calculated
as the mean of five `position_*` values in `candidates.csv`. A `misleading`
action must provide a different `message_position` in [-1, 1]. On its first
active day, an action selects a fixed, seeded uniform sample of constituency
voters. The same recipients receive the signal for the rest of its duration.

The implementation accepts `message_type: "ordinary"` and `"misleading"`,
but only `target_policy: "uniform"`. Group and network-based targeting are
the next campaign feature.
