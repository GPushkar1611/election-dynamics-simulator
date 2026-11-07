"""Campaign tests: timing, reach, ordinary and smear effects, reproducibility.

Run from the repository root:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from election_sim import SimulationConfig, simulate
from election_sim.campaign import CampaignAction, load_actions
from election_sim.campaign.effects import (BELIEVES, REJECTS, UNAWARE, apply_ordinary, apply_smear,
                                          assign_communities, message_position, select_recipients,
                                          spread_smears)
from test_baseline import fixture


def targeting_fixture():
    data = fixture()
    data["voters"] = data["voters"].copy()
    data["voters"]["occupation"] = ["Farmer", "Student", "Unemployed"]
    data["voters"]["urban_rural"] = ["Rural", "Urban", "Urban"]
    data["voters"]["wealth_level"] = ["Low", "Middle", "High"]
    return data


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
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="rumour")
        with self.assertRaisesRegex(ValueError, "distortion is only valid"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, distortion=0.5)
        with self.assertRaisesRegex(ValueError, "distortion must be in"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="misleading", distortion=2)
        with self.assertRaisesRegex(ValueError, "omit target_candidate_id"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, message_type="misleading",
                           target_candidate_id="right")
        with self.assertRaisesRegex(ValueError, "target_policy"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, target_policy="bogus")
        with self.assertRaisesRegex(ValueError, "target_group"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, target_policy="group")
        with self.assertRaisesRegex(ValueError, "target_group"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5,
                           target_policy="group", target_group="Unknown")
        with self.assertRaisesRegex(ValueError, "community_id"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5, community_id=0)
        with self.assertRaisesRegex(ValueError, "community_id"):
            CampaignAction("a", "left", 1, 1, 0.5, 0.5,
                           target_policy="community", community_id=-1)

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
        voters = targeting_fixture()["voters"]
        voters = pd.concat([voters, voters, voters, voters], ignore_index=True).iloc[:10]
        edges = targeting_fixture()["social_network"]
        for reach, size in ((0.0, 0), (0.3, 3), (1.0, 10)):
            chosen = select_recipients(CampaignAction("a", "left", 1, 1, 0.5, reach),
                                       voters, edges.iloc[:0], rng)
            self.assertEqual(len(chosen), size)
            self.assertEqual(len(set(chosen)), len(chosen))

    def test_same_seed_gives_same_recipients(self):
        action = CampaignAction("a", "left", 1, 1, 0.5, 0.3)
        data = targeting_fixture()
        np.testing.assert_array_equal(select_recipients(action, data["voters"], data["social_network"],
                                                        np.random.default_rng(7)),
                                      select_recipients(action, data["voters"], data["social_network"],
                                                        np.random.default_rng(7)))

    def test_group_targeting_maps_only_matching_voters_and_reports_shortfall(self):
        data = targeting_fixture()
        group_action = CampaignAction("group", "left", 1, 1, 0.5, 0.34,
                                      target_policy="group", target_group="Students")
        selected = select_recipients(group_action, data["voters"], data["social_network"],
                                     np.random.default_rng(1))
        np.testing.assert_array_equal(selected, [1])
        scarce = CampaignAction("scarce", "left", 1, 1, 0.5, 1.0,
                                target_policy="group", target_group="Unemployed")
        all_eligible = select_recipients(scarce, data["voters"], data["social_network"],
                                         np.random.default_rng(1))
        np.testing.assert_array_equal(all_eligible, [2])

    def test_all_documented_group_labels_map_to_voter_attributes(self):
        data = targeting_fixture()
        cases = {
            "General": [0, 1, 2], "Rural": [0], "Urban": [1, 2],
            "Students": [1], "Farmers": [0], "Unemployed": [2],
            "Low-income": [0], "Middle-income": [1], "High-income": [2],
        }
        for group, expected in cases.items():
            with self.subTest(group=group):
                action = CampaignAction(group, "left", 1, 1, 0.5, 1.0,
                                        target_policy="group", target_group=group)
                selected = select_recipients(action, data["voters"], data["social_network"],
                                             np.random.default_rng(1))
                np.testing.assert_array_equal(selected, expected)

    def test_high_susceptibility_selects_top_ranked_voter(self):
        data = targeting_fixture()
        data["voters"]["susceptibility"] = [0.1, 0.9, 0.5]
        action = CampaignAction("sus", "left", 1, 1, 0.5, 0.34,
                                target_policy="high_susceptibility")
        selected = select_recipients(action, data["voters"], data["social_network"],
                                     np.random.default_rng(1))
        np.testing.assert_array_equal(selected, [1])

    def test_high_influence_uses_weighted_outdegree_times_influence_power(self):
        data = targeting_fixture()
        data["voters"]["influence_power"] = [1.0, 2.0, 1.0]
        data["social_network"] = pd.DataFrame({
            "source_voter_id": ["a", "b"], "target_voter_id": ["b", "c"],
            "edge_weight": [1.0, 1.0],
        })
        action = CampaignAction("influence", "left", 1, 1, 0.5, 0.34,
                                target_policy="high_influence")
        selected = select_recipients(action, data["voters"], data["social_network"],
                                     np.random.default_rng(1))
        np.testing.assert_array_equal(selected, [1])

    def test_seeded_community_detection_and_target_selection_are_reproducible(self):
        voters = pd.DataFrame({"voter_id": [f"v{i}" for i in range(6)]})
        edges = pd.DataFrame({"source_voter_id": ["v0", "v2", "v4"],
                              "target_voter_id": ["v1", "v3", "v5"],
                              "edge_weight": [1.0, 1.0, 1.0]})
        labels = assign_communities(voters, edges, np.random.default_rng(13))
        again = assign_communities(voters, edges, np.random.default_rng(13))
        np.testing.assert_array_equal(labels, again)
        self.assertEqual(np.unique(labels).size, 3)
        action = CampaignAction("community", "left", 1, 1, 0.5, 0.34,
                                target_policy="community")
        first = select_recipients(action, voters, edges, np.random.default_rng(9), communities=labels)
        second = select_recipients(action, voters, edges, np.random.default_rng(9), communities=labels)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(np.unique(labels[first]).size, 1)

    def test_explicit_community_selection_uses_only_that_community(self):
        voters = pd.DataFrame({"voter_id": [f"v{i}" for i in range(6)]})
        edges = pd.DataFrame({"source_voter_id": ["v0", "v2", "v4"],
                              "target_voter_id": ["v1", "v3", "v5"],
                              "edge_weight": [1.0, 1.0, 1.0]})
        labels = assign_communities(voters, edges, np.random.default_rng(13))
        community_id = int(labels[2])
        action = CampaignAction("community", "left", 1, 1, 0.5, 1.0,
                                target_policy="community", community_id=community_id)
        selected = select_recipients(action, voters, edges, np.random.default_rng(9), communities=labels)
        np.testing.assert_array_equal(selected, np.flatnonzero(labels == community_id))

    def test_community_budget_shortfall_is_bounded_by_community_size(self):
        voters = pd.DataFrame({"voter_id": [f"v{i}" for i in range(6)]})
        edges = pd.DataFrame({"source_voter_id": ["v0", "v2", "v4"],
                              "target_voter_id": ["v1", "v3", "v5"],
                              "edge_weight": [1.0, 1.0, 1.0]})
        labels = assign_communities(voters, edges, np.random.default_rng(13))
        action = CampaignAction("community", "left", 1, 1, 0.5, 0.5,
                                target_policy="community", community_id=int(labels[0]))
        selected = select_recipients(action, voters, edges, np.random.default_rng(9), communities=labels)
        self.assertEqual(len(selected), 2)


class EffectTests(unittest.TestCase):
    def test_ordinary_moves_only_recipients_toward_platform(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5)
        opinions = apply_ordinary(np.array([0.0, 0.5]), np.array([1.0, 0.5]),
                                  np.array([1]), -1.0, action, 1.0)
        np.testing.assert_allclose(opinions, [0.0, -0.25])

    def test_misleading_position_moves_platform_toward_audience(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5, message_type="misleading", distortion=0.5)
        opinions = np.array([1.0, 0.6, -1.0])
        # Audience = voters 0 and 1 (mean 0.8); platform -1 moves halfway to 0.8.
        self.assertAlmostEqual(message_position(action, -1.0, opinions, np.array([0, 1])), -0.1)
        honest = CampaignAction("b", "left", 1, 1, 1.0, 0.5, message_type="misleading", distortion=0.0)
        self.assertEqual(message_position(honest, -1.0, opinions, np.array([0, 1])), -1.0)
        ordinary = CampaignAction("c", "left", 1, 1, 1.0, 0.5)
        self.assertEqual(message_position(ordinary, -1.0, opinions, np.array([0, 1])), -1.0)

    def test_smear_penalises_only_recipients_and_target(self):
        action = CampaignAction("a", "left", 1, 1, 1.0, 0.5, message_type="smear", target_candidate_id="right")
        penalty = apply_smear(np.zeros((2, 2)), np.array([1.0, 0.5]), np.array([1]), 1, action, 0.4)
        np.testing.assert_allclose(penalty, [[0.0, 0.0], [0.0, 0.2]])


class SmearSpreadTests(unittest.TestCase):
    """a -> b -> c chain; candidate column 0."""
    sources, targets = np.array([0, 1]), np.array([1, 2])

    def spread(self, belief, strength, weights=(1.0, 1.0), susceptibility=(1.0, 1.0, 1.0), p=1.0):
        return spread_smears(np.array(belief, dtype=np.int8).reshape(3, 1),
                             np.array(strength, dtype=float).reshape(3, 1), self.sources, self.targets,
                             np.array(weights), np.array(susceptibility), p, np.random.default_rng(0))

    def test_rumour_moves_one_tie_per_step(self):
        belief, strength = self.spread([BELIEVES, UNAWARE, UNAWARE], [0.4, 0, 0])
        np.testing.assert_array_equal(belief[:, 0], [BELIEVES, BELIEVES, UNAWARE])  # synchronous
        np.testing.assert_allclose(strength[:, 0], [0.4, 0.4, 0.0])

    def test_rumour_weakens_with_tie_weight(self):
        belief, strength = np.array([BELIEVES, UNAWARE, UNAWARE]), np.array([0.4, 0, 0])
        rng = np.random.default_rng(0)
        for _ in range(30):  # a weight-0.5 tie passes it on half the time
            belief, strength = spread_smears(belief.reshape(3, 1), strength.reshape(3, 1),
                                             self.sources, self.targets, np.array([0.5, 1.0]),
                                             np.ones(3), 1.0, rng)
            belief, strength = belief[:, 0], strength[:, 0]
        np.testing.assert_array_equal(belief, [BELIEVES, BELIEVES, BELIEVES])
        np.testing.assert_allclose(strength, [0.4, 0.2, 0.2])  # 0.4 x 0.5, then x 1.0

    def test_unsusceptible_listener_rejects_for_good(self):
        belief, _ = self.spread([BELIEVES, UNAWARE, UNAWARE], [0.4, 0, 0], susceptibility=(1, 0, 1))
        self.assertEqual(belief[1, 0], REJECTS)
        again, _ = self.spread(belief[:, 0], [0.4, 0, 0])
        self.assertEqual(again[1, 0], REJECTS)  # a rejecter neither re-believes nor passes it on
        self.assertEqual(again[2, 0], UNAWARE)

    def test_zero_probability_spreads_nothing(self):
        belief, strength = self.spread([BELIEVES, UNAWARE, UNAWARE], [0.4, 0, 0], p=0.0)
        np.testing.assert_array_equal(belief[:, 0], [BELIEVES, UNAWARE, UNAWARE])
        np.testing.assert_array_equal(strength[:, 0], [0.4, 0, 0])


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

    def test_simulation_records_group_audience_coverage_without_recipient_ids(self):
        data = targeting_fixture()
        action = CampaignAction("students", "left", 1, 1, 1.0, 0.34,
                                target_policy="group", target_group="Students")
        result = simulate(SimulationConfig(days=1, lambda_social=0, lambda_campaign=1), data, 42, [action])
        audience, = result.diagnostics["campaign_audiences"]
        self.assertEqual(audience["requested_recipients"], 1)
        self.assertEqual(audience["selected_recipients"], 1)
        self.assertEqual(audience["eligible_group_population"], 1)
        self.assertEqual(audience["shortfall"], 0)
        self.assertNotIn("voter_ids", audience)

    def test_simulation_reports_group_budget_shortfall(self):
        data = targeting_fixture()
        action = CampaignAction("students", "left", 1, 1, 1.0, 1.0,
                                target_policy="group", target_group="Students")
        result = simulate(SimulationConfig(days=1, lambda_social=0, lambda_campaign=1), data, 42, [action])
        audience, = result.diagnostics["campaign_audiences"]
        self.assertEqual(audience["requested_recipients"], 3)
        self.assertEqual(audience["selected_recipients"], 1)
        self.assertEqual(audience["shortfall"], 2)

    def test_simulation_assigns_communities_once_and_records_aggregate_sizes(self):
        data = targeting_fixture()
        action = CampaignAction("community", "left", 1, 1, 1.0, 0.34,
                                target_policy="community")
        result = simulate(SimulationConfig(days=1, lambda_social=0, lambda_campaign=1), data, 42, [action])
        audience, = result.diagnostics["campaign_audiences"]
        self.assertEqual(audience["selected_recipients"], 1)
        self.assertEqual(result.diagnostics["community_count"], 1)
        self.assertEqual(result.diagnostics["community_sizes"], [3])
        self.assertIn("community", result.metadata["child_seeds"])

    def test_smear_lowers_target_support_without_moving_opinions(self):
        control = simulate(SimulationConfig(days=1, lambda_social=0), fixture(), 42)
        smear = CampaignAction("a", "left", 1, 1, 1.0, 1.0, message_type="smear", target_candidate_id="right")
        result = run([smear])
        np.testing.assert_array_equal(control.final_state.final_opinion, result.final_state.final_opinion)
        self.assertTrue((support(result, "right") < support(control, "right")).all())
        self.assertEqual(result.metadata["scenario"], "smear_campaign")

    def test_misleading_with_zero_distortion_equals_ordinary(self):
        ordinary = run([CampaignAction("a", "left", 1, 1, 1.0, 0.67)])
        misleading = run([CampaignAction("a", "left", 1, 1, 1.0, 0.67,
                                         message_type="misleading", distortion=0.0)])
        np.testing.assert_array_equal(ordinary.final_state.final_opinion,
                                      misleading.final_state.final_opinion)
        self.assertEqual(misleading.metadata["scenario"], "misleading_campaign")

    def test_misleading_pulls_less_far_than_the_true_platform(self):
        # Everyone reached; audience mean is 0, so a distorted "left" message claims -0.5.
        ordinary = run([CampaignAction("a", "left", 1, 1, 1.0, 1.0)])
        misleading = run([CampaignAction("a", "left", 1, 1, 1.0, 1.0,
                                         message_type="misleading", distortion=0.5)])
        np.testing.assert_allclose(ordinary.final_state.final_opinion, [-1.0, -1.0, -1.0])
        np.testing.assert_allclose(misleading.final_state.final_opinion, [-0.5, -0.5, -0.5])

    def test_smear_spreads_only_when_enabled(self):
        # High influence picks voter b, who influences c (a -> b -> c).
        smear = CampaignAction("a", "left", 1, 1, 1.0, 0.34, message_type="smear",
                               target_candidate_id="right", target_policy="high_influence")
        quiet = simulate(SimulationConfig(days=3, lambda_social=0, lambda_campaign=1),
                         fixture(), 42, [smear])
        loud = simulate(SimulationConfig(days=3, lambda_social=0, lambda_campaign=1,
                                         smear_spread_probability=1.0), fixture(), 42, [smear])
        self.assertEqual(quiet.diagnostics["smear_believers"]["right"], 1)
        spread = loud.diagnostics["smear_believers"]["right"] + loud.diagnostics["smear_rejecters"]["right"]
        self.assertGreater(spread, 1)
        believers = loud.daily_summary.smear_believers.to_numpy()
        self.assertTrue((np.diff(believers) >= 0).all())  # nobody forgets in this model

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
