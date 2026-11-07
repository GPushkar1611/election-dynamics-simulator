"""Small paired counterfactuals for supplied news/event schedules."""

from dataclasses import dataclass, replace

import pandas as pd

from .simulation import RunResult, simulate


@dataclass
class PairedEventResult:
    no_shock: RunResult
    with_shock: RunResult
    candidate_effects: pd.DataFrame
    opinion_effect: dict


def compare_event_shock(config, data: dict, seed: int = 42,
                        campaign_actions=None, agents=None) -> PairedEventResult:
    """Run matched no-event and event scenarios with common random streams.

    The only changed parameter is ``lambda_event``. Social, campaign, turnout,
    voting, event occurrence, and event reminder streams use the same seeds.
    The supplied schedule is retained in both runs; the control has zero event
    strength.
    """
    if config.lambda_event <= 0:
        raise ValueError("the shock run requires lambda_event > 0")
    no_shock = simulate(replace(config, lambda_event=0.0), data, seed, campaign_actions, agents=agents)
    with_shock = simulate(config, data, seed, campaign_actions, agents=agents)
    day = config.days
    control_support = no_shock.daily_support.query("day == @day").set_index("candidate_id").mean_support
    shock_support = with_shock.daily_support.query("day == @day").set_index("candidate_id").mean_support
    control_votes = no_shock.candidate_results.set_index("candidate_id").vote_share
    shock_votes = with_shock.candidate_results.set_index("candidate_id").vote_share
    candidate_effects = pd.DataFrame({
        "candidate_id": control_support.index,
        "support_no_shock": control_support.to_numpy(),
        "support_with_shock": shock_support.reindex(control_support.index).to_numpy(),
        "support_difference": (shock_support - control_support).reindex(control_support.index).to_numpy(),
        "vote_share_no_shock": control_votes.reindex(control_support.index).to_numpy(),
        "vote_share_with_shock": shock_votes.reindex(control_support.index).to_numpy(),
        "vote_share_difference": (shock_votes - control_votes).reindex(control_support.index).to_numpy(),
    })
    opinion_delta = (with_shock.final_state.final_opinion.to_numpy()
                     - no_shock.final_state.final_opinion.to_numpy())
    opinion_effect = {"mean_difference": float(opinion_delta.mean()),
                      "mean_absolute_difference": float(abs(opinion_delta).mean()),
                      "voters_changed": int((opinion_delta != 0).sum()),
                      "voter_count": len(opinion_delta)}
    return PairedEventResult(no_shock, with_shock, candidate_effects, opinion_effect)
