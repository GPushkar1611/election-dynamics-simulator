"""Strategic voting after public polls (epistemic logic; see election_sim/strategic.py).

Run from the repository root:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import unittest

import numpy as np
import pandas as pd

from election_sim import CandidateAgent, NewsAgent, SimulationConfig, simulate
from election_sim.strategic import strategic_adjustment, viable_candidates
from election_sim.support import ISSUES
from test_agents import crowd
from test_events import event

POLL_DAY = 2


def three_way_race():
    """40 voters from -1 to 1; "left" (-1) and "right" (+1) split the wings,
    a weaker "centre" (0) wins only the middle (about 6 of 40 voters)."""
    data = crowd()
    data["voters"]["event_sensitivity"] = 1.0
    data["voters"]["region_id"] = 1
    centre = pd.DataFrame({"candidate_id": ["centre"], "constituency_id": [1],
                           "local_popularity": [0.4],
                           **{f"position_{issue}": [0.0] for issue in ISSUES}})
    data["candidates"] = pd.concat([data["candidates"], centre], ignore_index=True)
    return data


def public_poll(day=POLL_DAY):
    return NewsAgent(event(event_id="poll", event_type="poll_release", event_day=day,
                           candidate_id=None, candidate_effect=None))


def run(strategic_voting=0.0, news=None, data=None, **kwargs):
    config = SimulationConfig(days=4, lambda_social=0, poll_sample_size=40, poll_noise=0,
                              strategic_voting=strategic_voting)
    return simulate(config, data or three_way_race(), 42, record_history=True,
                    news_agent=news or NewsAgent(pd.DataFrame()), **kwargs)


def top_choice_counts(result, day):
    leaders = result.history["support"][day].argmax(axis=1)
    return dict(zip(result.history["candidate_ids"], np.bincount(leaders, minlength=3)))


class ViabilityTests(unittest.TestCase):
    def test_top_two_in_the_poll_are_viable(self):
        ids = ["a", "b", "c"]
        np.testing.assert_array_equal(viable_candidates({"a": 0.2, "b": 0.5, "c": 0.3}, ids),
                                      [False, True, True])

    def test_tie_at_the_cutoff_goes_to_the_earlier_id(self):
        ids = ["a", "b", "c"]
        np.testing.assert_array_equal(viable_candidates({"a": 0.25, "b": 0.5, "c": 0.25}, ids),
                                      [True, True, False])

    def test_two_candidates_are_always_both_viable(self):
        self.assertTrue(viable_candidates({"a": 0.9, "b": 0.1}, ["a", "b"]).all())

    def test_adjustment_penalises_only_non_viable(self):
        np.testing.assert_array_equal(strategic_adjustment(np.array([True, False, True]), 1.5),
                                      [0.0, -1.5, 0.0])


class StrategicVotingTests(unittest.TestCase):
    def test_public_poll_makes_supporters_of_a_hopeless_candidate_switch(self):
        sincere = run(0.0, public_poll())
        strategic = run(2.0, public_poll())
        poll, = strategic.diagnostics["public_polls"]
        self.assertEqual(poll["non_viable"], ["centre"])
        # Identical until the poll is published...
        np.testing.assert_array_equal(sincere.history["support"][:POLL_DAY],
                                      strategic.history["support"][:POLL_DAY])
        # ...then every centre supporter switches to a front-runner.
        before = top_choice_counts(sincere, POLL_DAY)["centre"]
        self.assertGreater(before, 0)
        self.assertEqual(top_choice_counts(strategic, POLL_DAY)["centre"], 0)
        self.assertEqual(poll["switched_top_choice"], before)
        self.assertLess(strategic.final_state.support_centre.mean(),
                        sincere.final_state.support_centre.mean())

    def test_weak_strategic_pull_keeps_committed_supporters(self):
        weak = run(0.1, public_poll())
        self.assertGreater(top_choice_counts(weak, POLL_DAY)["centre"], 0)

    def test_without_strategic_voting_a_public_poll_changes_nothing(self):
        control = run(0.0)
        polled = run(0.0, public_poll())
        np.testing.assert_array_equal(control.history["support"], polled.history["support"])
        self.assertNotIn("non_viable", polled.diagnostics["public_polls"][0])

    def test_private_agent_polls_do_not_trigger_strategic_voting(self):
        # Agents poll privately; with no public announcement voters know nothing.
        config = SimulationConfig(days=4, lambda_social=0, lambda_campaign=1, poll_noise=0)
        agent = [CandidateAgent("left", budget=0)]
        sincere = simulate(config, three_way_race(), 42, agents=agent)
        strategic = simulate(SimulationConfig(**{**config.__dict__, "strategic_voting": 2.0}),
                             three_way_race(), 42, agents=agent)
        self.assertTrue(sincere.diagnostics["polls"])
        pd.testing.assert_frame_equal(sincere.final_state, strategic.final_state)

    def test_two_candidate_race_is_unaffected(self):
        data = three_way_race()
        data["candidates"] = data["candidates"].iloc[:2]
        sincere = run(0.0, public_poll(), data)
        strategic = run(2.0, public_poll(), data)
        np.testing.assert_array_equal(sincere.history["support"], strategic.history["support"])
        self.assertEqual(strategic.diagnostics["public_polls"][0]["switched_top_choice"], 0)

    def test_zero_event_strength_disables_the_announcement(self):
        config = SimulationConfig(days=4, lambda_social=0, lambda_event=0, strategic_voting=2.0)
        result = simulate(config, three_way_race(), 42, news_agent=public_poll())
        self.assertEqual(result.diagnostics["public_polls"], [])

    def test_invalid_strength_is_rejected(self):
        with self.assertRaises(ValueError):
            SimulationConfig(strategic_voting=-1)


if __name__ == "__main__":
    unittest.main()
