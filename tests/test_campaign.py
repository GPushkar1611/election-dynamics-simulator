import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from election_sim import SimulationConfig, simulate
from election_sim.campaign import CampaignAction, load_actions
from election_sim.campaign.effects import apply_campaign_action, select_uniform_recipients
from test_baseline import fixture


class CampaignActionTests(unittest.TestCase):
    def test_action_calendar_is_inclusive_at_start_and_exclusive_after_duration(self):
        action = CampaignAction("a", "left", 10, 2, 0.5, 0.5)
        self.assertFalse(action.active_on(9))
        self.assertTrue(action.active_on(10))
        self.assertTrue(action.active_on(11))
        self.assertFalse(action.active_on(12))

    def test_uniform_selection_is_seeded_unique_and_has_declared_size(self):
        first = select_uniform_recipients(10, 0.3, np.random.default_rng(20))
        second = select_uniform_recipients(10, 0.3, np.random.default_rng(20))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_campaign_message_moves_only_recipients_toward_message_position(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5)
        opinions = apply_campaign_action(np.array([0.0, 0.5]), np.array([1.0, 0.5]),
                                         np.array([1]), -1.0, action, 1.0)
        np.testing.assert_allclose(opinions, [0.0, -0.25])

    def test_invalid_action_type_or_target_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "require message_position"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="misleading")
        with self.assertRaisesRegex(ValueError, "omit message_position"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_position=0.0)
        with self.assertRaisesRegex(ValueError, "target_policy"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, target_policy="high_influence")

    def test_misleading_action_uses_declared_message_position(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 1.0,
                                message_type="misleading", message_position=0.5)
        opinions = apply_campaign_action(np.array([0.0]), np.array([1.0]),
                                         np.array([0]), 0.5, action, 1.0)
        np.testing.assert_allclose(opinions, [0.5])

    def test_loader_rejects_unknown_candidate_and_too_late_action(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            path.write_text(json.dumps({"actions": [{"action_id": "a", "candidate_id": "missing",
                              "start_day": 2, "duration": 1, "intensity": 0.5,
                              "reach_fraction": 0.5}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown candidate"):
                load_actions(path, 2, {"left"})
            path.write_text(json.dumps({"actions": [{"action_id": "a", "candidate_id": "left",
                              "start_day": 2, "duration": 2, "intensity": 0.5,
                              "reach_fraction": 0.5}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "beyond"):
                load_actions(path, 2, {"left"})


class CampaignSimulationTests(unittest.TestCase):
    def test_full_reach_campaign_changes_opinion_and_is_recorded(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 1.0)
        result = simulate(SimulationConfig(days=1, lambda_social=0, lambda_campaign=1),
                          fixture(), 42, [action])
        np.testing.assert_allclose(result.final_state.final_opinion, [-1.0, -1.0, -1.0])
        self.assertEqual(result.metadata["scenario"], "ordinary_campaign")
        self.assertEqual(result.diagnostics["campaign_actions_loaded"], 1)
        self.assertEqual(result.diagnostics["campaign_exposures"], 3)
        self.assertEqual(result.diagnostics["campaigns_applied"], 1)

    def test_zero_campaign_strength_is_the_existing_social_control(self):
        action = CampaignAction("a", "left", 1, 2, 1.0, 1.0)
        data = fixture()
        without = simulate(SimulationConfig(days=2, lambda_social=0, lambda_campaign=0), data, 42)
        with_action = simulate(SimulationConfig(days=2, lambda_social=0, lambda_campaign=0), data, 42, [action])
        np.testing.assert_array_equal(without.final_state.final_opinion, with_action.final_state.final_opinion)


if __name__ == "__main__":
    unittest.main()
