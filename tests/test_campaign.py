"""Campaign tests: timing, reach, ordinary and smear effects, reproducibility.

Run from the repository root:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from election_sim import SimulationConfig, simulate
from election_sim.campaign import CampaignAction, load_actions
from election_sim.campaign.effects import apply_ordinary, apply_smear, select_recipients
from test_baseline import fixture


def run(actions, days=1, lambda_campaign=1.0):
    config = SimulationConfig(days=days, lambda_social=0, lambda_campaign=lambda_campaign)
    return simulate(config, fixture(), 42, actions)


def support(result, candidate):
    return result.final_state[f"support_{candidate}"].to_numpy()


class ActionTests(unittest.TestCase):
    def test_active_from_start_day_for_duration_days(self):
        action = CampaignAction("a", "left", 10, 2, 0.5, 0.5)
        self.assertEqual([action.active_on(day) for day in (9, 10, 11, 12)],
                         [False, True, True, False])

    def test_invalid_actions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "require target_candidate_id"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="smear")
        with self.assertRaisesRegex(ValueError, "cannot target its own"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="smear", target_candidate_id="left")
        with self.assertRaisesRegex(ValueError, "omit target_candidate_id"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, target_candidate_id="right")
        with self.assertRaisesRegex(ValueError, "message_type"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="misleading")
        with self.assertRaisesRegex(ValueError, "target_policy"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, target_policy="high_influence")

    def test_loader_rejects_unknown_candidates_and_late_actions(self):
        base = {"action_id": "a", "candidate_id": "left", "start_day": 2, "duration": 1,
                "intensity": 0.5, "reach_fraction": 0.5}
        bad = [({**base, "candidate_id": "missing"}, "unknown candidate"),
               ({**base, "message_type": "smear", "target_candidate_id": "missing"}, "unknown candidate"),
               ({**base, "duration": 2}, "beyond")]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            for item, message in bad:
                path.write_text(json.dumps({"actions": [item]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_actions(path, 2, {"left", "right"})


class RecipientTests(unittest.TestCase):
    def test_uniform_selection_size_zero_and_full_reach(self):
        rng = np.random.default_rng(0)
        for reach, size in ((0.0, 0), (0.3, 3), (1.0, 10)):
            chosen = select_recipients(CampaignAction("a", "left", 1, 1, 0.5, reach), 10, rng)
            self.assertEqual(len(chosen), size)
            self.assertEqual(len(set(chosen)), size)

    def test_same_seed_gives_same_recipients(self):
        action = CampaignAction("a", "left", 1, 1, 0.5, 0.3)
        np.testing.assert_array_equal(select_recipients(action, 10, np.random.default_rng(7)),
                                      select_recipients(action, 10, np.random.default_rng(7)))


class EffectTests(unittest.TestCase):
    def test_ordinary_moves_only_recipients_toward_platform(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5)
        opinions = apply_ordinary(np.array([0.0, 0.5]), np.array([1.0, 0.5]),
                                  np.array([1]), -1.0, action, 1.0)
        np.testing.assert_allclose(opinions, [0.0, -0.25])

    def test_smear_penalises_only_recipients_and_target(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5, message_type="smear", target_candidate_id="right")
        penalty = apply_smear(np.zeros((2, 2)), np.array([1.0, 0.5]), np.array([1]), 1, action, 0.4)
        np.testing.assert_allclose(penalty, [[0.0, 0.0], [0.0, 0.2]])


class SimulationTests(unittest.TestCase):
    def test_zero_reach_changes_nobody(self):
        control = simulate(SimulationConfig(days=1, lambda_social=0), fixture(), 42)
        result = run([CampaignAction("a", "left", 1, 1, 1.0, 0.0)])
        np.testing.assert_array_equal(control.final_state.final_opinion, result.final_state.final_opinion)
        np.testing.assert_array_equal(support(control, "right"), support(result, "right"))

    def test_full_reach_ordinary_moves_everyone_to_platform(self):
        result = run([CampaignAction("a", "left", 1, 1, 1.0, 1.0)])
        np.testing.assert_allclose(result.final_state.final_opinion, [-1.0, -1.0, -1.0])
        self.assertEqual(result.metadata["scenario"], "ordinary_campaign")
        self.assertEqual(result.diagnostics["campaign_exposures"], 3)

    def test_smear_lowers_target_support_without_moving_opinions(self):
        control = simulate(SimulationConfig(days=1, lambda_social=0), fixture(), 42)
        smear = CampaignAction("a", "left", 1, 1, 1.0, 1.0, message_type="smear", target_candidate_id="right")
        result = run([smear])
        np.testing.assert_array_equal(control.final_state.final_opinion, result.final_state.final_opinion)
        self.assertTrue((support(result, "right") < support(control, "right")).all())
        self.assertEqual(result.metadata["scenario"], "smear_campaign")

    def test_actions_without_campaign_strength_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "lambda_campaign is 0"):
            run([CampaignAction("a", "left", 1, 1, 1.0, 1.0)], lambda_campaign=0)

    def test_adding_or_reordering_actions_keeps_other_recipients(self):
        ordinary = CampaignAction("a", "left", 1, 2, 1.0, 0.34)
        smear = CampaignAction("b", "left", 1, 2, 1.0, 0.34, message_type="smear", target_candidate_id="right")
        alone = run([ordinary], days=2).final_state.final_opinion
        # A smear never moves opinions, so ordinary's effect must be unchanged.
        np.testing.assert_array_equal(alone, run([ordinary, smear], days=2).final_state.final_opinion)
        np.testing.assert_array_equal(alone, run([smear, ordinary], days=2).final_state.final_opinion)


if __name__ == "__main__":
    unittest.main()
