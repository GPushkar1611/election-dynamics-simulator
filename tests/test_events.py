"""Scheduled event timing, audience, and signal semantics."""

import unittest

import numpy as np
import pandas as pd

from election_sim import SimulationConfig, simulate
from election_sim.experiments import compare_event_shock
from test_baseline import fixture


def event(**overrides):
    row = {"event_id": "event-1", "event_day": 1, "duration": 1,
           "source_type": "News", "event_type": "Policy announcement",
           "issue": "general", "candidate_id": "left", "intensity": 1.0,
           "credibility": 1.0, "target_group": "General", "candidate_effect": 1.0,
           "region_id": 1, "constituency_id": 1}
    row.update(overrides)
    return pd.DataFrame([row])


def event_fixture():
    data = fixture()
    voters = data["voters"].copy()
    voters["event_sensitivity"] = [1.0, 0.5, 0.0]
    voters["constituency_id"] = 1
    voters["region_id"] = 1
    voters["occupation"] = ["Farmer", "Student", "Unemployed"]
    voters["urban_rural"] = ["Rural", "Urban", "Rural"]
    voters["wealth_level"] = ["Low", "Middle", "High"]
    for issue in ("employment", "economy", "education", "healthcare", "agriculture"):
        voters[f"issue_{issue}"] = 0.2
    data["voters"] = voters
    return data


def run(data, events, days=3, strength=1.0):
    data = {**data, "events": events}
    return simulate(SimulationConfig(days=days, lambda_social=0, lambda_event=strength), data, 17)


class EventSimulationTests(unittest.TestCase):
    def test_candidate_signal_timing_and_persists_after_event(self):
        data = event_fixture()
        control = run(data, pd.DataFrame(), days=3)
        result = run(data, event(event_day=2), days=3)
        base = control.daily_support.query("candidate_id == 'left'").mean_support.to_numpy()
        changed = result.daily_support.query("candidate_id == 'left'").mean_support.to_numpy()
        self.assertAlmostEqual(changed[0], base[0])
        self.assertAlmostEqual(changed[1], base[1])
        self.assertGreater(changed[2], base[2])
        self.assertGreater(changed[2], changed[3])
        self.assertGreater(changed[3], base[3])
        self.assertEqual(result.diagnostics["events_applied"], 1)

    def test_duration_spreads_total_candidate_signal_over_active_days(self):
        data = event_fixture()
        one_day = run(data, event(event_day=1, duration=1), days=2)
        two_days = run(data, event(event_day=1, duration=2), days=2)
        a = one_day.daily_support.query("day == 1 and candidate_id == 'left'").mean_support.iloc[0]
        b = two_days.daily_support.query("day == 2 and candidate_id == 'left'").mean_support.iloc[0]
        self.assertAlmostEqual(a, b)
        self.assertEqual(two_days.diagnostics["events_applied"], 2)

    def test_group_issue_and_sensitivity_limit_candidate_effect_audience(self):
        data = event_fixture()
        result = run(data, event(target_group="Rural", issue="employment"), days=1)
        final = result.final_state
        control = run(data, pd.DataFrame(), days=1).final_state
        self.assertNotAlmostEqual(final.loc[0, "support_left"], control.loc[0, "support_left"])
        self.assertAlmostEqual(final.loc[1, "support_left"], control.loc[1, "support_left"])
        self.assertAlmostEqual(final.loc[2, "support_left"], control.loc[2, "support_left"])

    def test_neutral_event_moves_opinion_without_candidate_utility_signal(self):
        data = event_fixture()
        data["voters"]["initial_opinion"] = [-0.1, 0.0, 0.1]
        data["voters"]["event_sensitivity"] = 1.0
        neutral = event(candidate_id=None, candidate_effect=None, event_day=1)
        result = run(data, neutral, days=1)
        opinions = result.final_state.final_opinion.to_numpy()
        np.testing.assert_allclose(opinions.mean(), 0.0, atol=1e-12)
        self.assertGreater(np.abs(opinions - data["voters"].initial_opinion.to_numpy()).sum(), 0)
        self.assertIsNone(result.diagnostics["event_applications"][0]["candidate_id"])

    def test_directional_event_moves_the_audience_toward_configured_axis(self):
        data = event_fixture()
        data["voters"]["initial_opinion"] = [-0.1, 0.0, 0.1]
        data["voters"]["event_sensitivity"] = 1.0
        directional = event(candidate_id=None, candidate_effect=None, shock_mode="directional",
                            opinion_direction=0.75)
        result = run(data, directional, days=1)
        final = result.final_state.final_opinion.to_numpy()
        self.assertGreater(final.mean(), data["voters"].initial_opinion.mean())
        self.assertAlmostEqual(final.mean() - data["voters"].initial_opinion.mean(), 0.75)

    def test_zero_strength_is_a_no_shock_control(self):
        data = event_fixture()
        control = run(data, pd.DataFrame(), days=1)
        no_shock = run(data, event(), days=1, strength=0)
        np.testing.assert_allclose(no_shock.final_state.select_dtypes("number"),
                                   control.final_state.select_dtypes("number"))
        self.assertEqual(no_shock.diagnostics["events_loaded"], 1)
        self.assertEqual(no_shock.diagnostics["events_applied"], 0)
        self.assertEqual(no_shock.diagnostics["loaded_events_not_applied"], 1)

    def test_only_constituency_region_and_state_events_are_loaded(self):
        data = event_fixture()
        events = pd.concat([
            event(event_id="local"), event(event_id="regional", constituency_id=None),
            event(event_id="state", constituency_id=None, region_id=None),
            event(event_id="other-local", constituency_id=2),
            event(event_id="other-region", constituency_id=None, region_id=2),
        ], ignore_index=True)
        result = run(data, events, days=1)
        self.assertEqual(result.diagnostics["events_loaded"], 3)
        self.assertEqual({row["event_id"] for row in result.diagnostics["event_applications"]},
                         {"local", "regional", "state"})

    def test_zero_half_life_preserves_effect_while_default_memory_decays(self):
        data = event_fixture()
        permanent = run(data, event(memory_half_life_days=0), days=4)
        decaying = run(data, event(), days=4)
        permanent_support = permanent.daily_support.query("candidate_id == 'left'").mean_support.to_numpy()
        decaying_support = decaying.daily_support.query("candidate_id == 'left'").mean_support.to_numpy()
        self.assertAlmostEqual(permanent_support[1], permanent_support[4])
        self.assertGreater(decaying_support[1], decaying_support[2])
        self.assertGreater(decaying_support[2], decaying_support[4])

    def test_reminders_reinforce_an_event_after_its_active_duration(self):
        data = event_fixture()
        result = simulate(SimulationConfig(days=3, lambda_social=0, lambda_event=1,
                                           event_memory_half_life_days=0,
                                           event_reminder_probability=1),
                          {**data, "events": event()}, 17)
        self.assertEqual(result.diagnostics["event_reminders"], 2)
        self.assertGreater(result.diagnostics["events_applied"], 0)

    def test_conditional_event_is_suppressed_when_its_parent_does_not_occur(self):
        data = event_fixture()
        root = event(event_id="root", event_day=1, occurrence_probability=0)
        child = event(event_id="child", event_day=2, depends_on_event_id="root",
                      condition_probability=1)
        result = run(data, pd.concat([root, child], ignore_index=True), days=3)
        self.assertEqual(result.diagnostics["events_occurred"], 0)
        self.assertEqual(result.diagnostics["events_applied"], 0)
        self.assertEqual(result.diagnostics["loaded_events_not_applied"], 2)

    def test_candidate_withdrawal_removes_support_and_votes(self):
        data = event_fixture()
        withdrawal = event(event_type="Candidate withdrawal", structural_effect="candidate_withdrawal")
        result = run(data, withdrawal, days=1)
        self.assertEqual(result.diagnostics["withdrawn_candidates"], ["left"])
        self.assertTrue((result.final_state.support_left < 1e-10).all())
        left_votes = result.candidate_results.set_index("candidate_id").loc["left", "votes"]
        self.assertEqual(left_votes, 0)

    def test_repeated_withdrawal_of_same_candidate_does_not_block_remaining_candidate(self):
        data = event_fixture()
        first = event(event_id="withdraw-1", event_day=1,
                      event_type="Candidate withdrawal", structural_effect="candidate_withdrawal")
        second = event(event_id="withdraw-2", event_day=2,
                       event_type="Candidate withdrawal", structural_effect="candidate_withdrawal")
        result = run(data, pd.concat([first, second], ignore_index=True), days=2)
        self.assertEqual(result.diagnostics["withdrawn_candidates"], ["left"])
        self.assertEqual(len(result.candidate_results), 2)

    def test_paired_shock_experiment_uses_common_seed_streams(self):
        data = {**event_fixture(), "events": event()}
        paired = compare_event_shock(SimulationConfig(days=2, lambda_social=0, lambda_event=0.8),
                                     data, seed=17)
        self.assertEqual(paired.no_shock.metadata["child_seeds"],
                         paired.with_shock.metadata["child_seeds"])
        self.assertGreater(paired.candidate_effects.support_difference.abs().sum(), 0)
        self.assertEqual(len(paired.candidate_effects), 2)


if __name__ == "__main__":
    unittest.main()
