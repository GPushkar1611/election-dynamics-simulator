# Model guide

This guide summarizes what the simulator computes. Parameters are scenario assumptions unless explicitly stated otherwise; the supplied data is synthetic.

## Voters and social influence

Each voter has an opinion `x` between -1 and 1. Every day, a voter averages opinions from incoming neighbors whose opinions are within that voter's confidence threshold. Edge weight and sender influence determine the average. The voter moves toward it according to susceptibility and `lambda_social`. All voters use the same start-of-day opinion snapshot.

This is a DeGroot-style consensus update with bounded confidence. It can converge, but bounded confidence can also leave separate groups. It is not one of the robotics controllers (rendezvous, flocking, formation control) from the MAS Algorithms lecture.

## Candidate support and voting

In the default scalar mode, candidate utility combines a popularity term and distance from the voter's opinion to the average of the candidate's five issue positions. A softmax converts utility into support probabilities. At election close, voters turn out according to their propensity (optionally adjusted by exposure) and either sample a candidate from their support or choose their highest-support candidate.

The optional issue-profile mode synthesizes voter positions on each issue because the data supplies issue importance, not measured voter positions. Treat it as sensitivity analysis.

## Campaigns and candidate agents

Ordinary and misleading messages move reached voters toward a candidate signal. Misleading messages shift the signal toward the reached audience's average opinion. Smears instead create a negative claim about another candidate and lower that candidate's utility for believers.

Smear claims can spread to social neighbors. Each voter is unaware, believes, or rejects a claim. Rejection is permanent in the current model; there is no recovery or reconsideration.

Candidate agents receive a poll shared with all agents who make decisions that day. Voters do not see that poll unless a public `poll_release` event occurs. The `rebut` strategy name is historical: it responds to a drop in the agent's poll share, not to verified knowledge of a smear.

## Events and public polls

Scheduled events can affect named candidates or apply neutral opinion shocks. Event signals can persist, decay, and receive reminders. Some schedules also model dependencies and candidate withdrawal. `lambda_event=0` disables event effects, including news announcements.

A public poll release is treated as received by everyone and updates voter support using a simplified strategic-voting rule: candidates outside the published poll's top two receive a utility penalty. This is an epistemic-logic-inspired assumption, not a formal knowledge model: the simulator does not represent possible worlds or individual knowledge states.

## Regional runs

Constituencies keep local ballots and campaigns. Social opinions can cross constituency boundaries after cross-constituency edge weights are scaled by the configured spillover value. A nonzero scale changes relative influence after incoming-weight normalization; it is not a direct multiplier on opinion movement.

## Daily order

For each simulated day: agents observe the previous support and choose actions; voters update socially from a common snapshot; campaigns apply; rumor claims spread; events and announcements apply; support is recalculated and history is recorded. Turnout and votes are sampled after the last day.

## What to call the course connections

- **Consensus theory:** DeGroot-style opinion averaging with bounded confidence.
- **Infection/contagion:** simple one-way rumor spread; no recovery.
- **MAS control:** heuristic candidate agents choose campaigns; robotics controllers are not implemented.
- **Epistemic logic:** public versus agent-only polls motivate a simplified rule; no formal epistemic model is implemented.

For every topic, describe the project as a simulation under assumptions, not as an empirically validated behavioral model.
