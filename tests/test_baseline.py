import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from election_sim import SimulationConfig, simulate
from election_sim.dynamics import social_step
from election_sim.network import build_incoming
from election_sim.support import ISSUES, candidate_support
from election_sim.voting import election_result


def fixture():
    voters = pd.DataFrame({
        "voter_id": ["a", "b", "c"], "constituency_id": [1] * 3,
        "initial_opinion": [-1., 0., 1.], "susceptibility": [1.] * 3,
        "confidence_threshold": [2.] * 3, "influence_power": [1., 2., 1.],
        "turnout_propensity": [0.7] * 3,
    })
    candidates = pd.DataFrame({
        "candidate_id": ["left", "right"], "constituency_id": [1, 1],
        "local_popularity": [0.5, 0.5],
        **{f"position_{issue}": [-1., 1.] for issue in ISSUES},
    })
    edges = pd.DataFrame({"source_voter_id": ["a", "b"],
                          "target_voter_id": ["b", "c"], "edge_weight": [1., 1.]})
    return {"voters": voters, "candidates": candidates, "social_network": edges}


class DynamicsTests(unittest.TestCase):
    def step(self, data, strength=0.5):
        voters = data["voters"]
        graph = build_incoming(voters, data["social_network"], 1.)
        return social_step(voters.initial_opinion.to_numpy(), voters.susceptibility.to_numpy(),
                           voters.confidence_threshold.to_numpy(), graph, strength)

    def test_one_edge_only_changes_target(self):
        data = fixture()
        data["social_network"] = data["social_network"].iloc[:1]
        opinions, count = self.step(data)
        np.testing.assert_allclose(opinions, [-1., -0.5, 1.])
        self.assertEqual(count, 1)

    def test_chain_uses_synchronous_snapshot(self):
        opinions, _ = self.step(fixture())
        np.testing.assert_allclose(opinions, [-1., -0.5, 0.5])

    def test_no_eligible_neighbors_no_movement(self):
        data = fixture()
        data["voters"]["confidence_threshold"] = 0.
        opinions, count = self.step(data)
        np.testing.assert_array_equal(opinions, [-1., 0., 1.])
        self.assertEqual(count, 0)

    def test_confidence_boundary_is_inclusive_and_target_specific(self):
        data = fixture()
        data["voters"]["confidence_threshold"] = [0., 1., 0.99]
        opinions, count = self.step(data)
        np.testing.assert_allclose(opinions, [-1., -0.5, 1.])
        self.assertEqual(count, 1)

    def test_zero_strength_and_zero_susceptibility(self):
        data = fixture()
        opinions, _ = self.step(data, 0.)
        np.testing.assert_array_equal(opinions, [-1., 0., 1.])
        data["voters"]["susceptibility"] = 0.
        np.testing.assert_array_equal(self.step(data)[0], [-1., 0., 1.])

    def test_sender_power_and_edge_weight_determine_mean(self):
        data = fixture()
        data["social_network"] = pd.DataFrame({"source_voter_id": ["a", "c"],
                                               "target_voter_id": ["b", "b"],
                                               "edge_weight": [1., 0.5]})
        data["voters"]["influence_power"] = [2., 1., 1.]
        np.testing.assert_allclose(self.step(data, 1.)[0], [-1., -0.6, 1.])

    def test_empty_graph_keeps_all_opinions(self):
        data = fixture()
        data["social_network"] = data["social_network"].iloc[:0]
        np.testing.assert_array_equal(self.step(data)[0], [-1., 0., 1.])

    def test_rho_zero_uses_tie_weights_without_sender_power(self):
        data = fixture()
        edges = pd.DataFrame({"source_voter_id": ["a", "c"],
                              "target_voter_id": ["b", "b"], "edge_weight": [1., 0.5]})
        data["voters"]["influence_power"] = [5., 1., 1.]
        voters = data["voters"]
        graph = build_incoming(voters, edges, 0.)
        opinions, _ = social_step(voters.initial_opinion.to_numpy(), np.ones(3),
                                   np.full(3, 2.), graph, 1.)
        np.testing.assert_allclose(opinions, [-1., -1 / 3, 1.])

    def test_invalid_graphs_are_rejected(self):
        for source, target in (("unknown", "a"), ("a", "a")):
            data = fixture()
            data["social_network"] = pd.DataFrame({"source_voter_id": [source],
                                                   "target_voter_id": [target], "edge_weight": [1.]})
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                self.step(data)


class SupportTests(unittest.TestCase):
    def test_hand_computed_support(self):
        candidates = fixture()["candidates"]
        support = candidate_support(np.array([-1., 0., 1.]), candidates)
        expected = 1 / (1 + np.exp(-2))
        np.testing.assert_allclose(support, [[expected, 1 - expected], [0.5, 0.5],
                                             [1 - expected, expected]])

    def test_popularity_prior_hand_example(self):
        candidates = fixture()["candidates"]
        candidates["local_popularity"] = [0.8, 0.5]
        np.testing.assert_allclose(candidate_support(np.array([0.]), candidates), [[0.8, 0.2]])

    def test_extreme_popularity_and_temperature_stay_finite(self):
        candidates = fixture()["candidates"]
        candidates["local_popularity"] = [0., 1.]
        support = candidate_support(np.array([-1., 0., 1.]), candidates, tau=1e-300)
        self.assertTrue(np.isfinite(support).all())
        np.testing.assert_allclose(support.sum(axis=1), 1.)

    def test_mean_ideology_uses_all_five_issues(self):
        candidates = fixture()["candidates"]
        candidates.loc[0, [f"position_{issue}" for issue in ISSUES]] = [-1., -1., 1., 1., 0.]
        candidates.loc[1, [f"position_{issue}" for issue in ISSUES]] = 0.
        np.testing.assert_allclose(candidate_support(np.array([0.]), candidates), [[0.5, 0.5]])


class VotingTests(unittest.TestCase):
    def vote(self, propensity, rule="sample"):
        support = np.tile([0.75, 0.25], (len(propensity), 1))
        return election_result(np.array(propensity), support, ["left", "right"],
                               np.random.default_rng(1), np.random.default_rng(2), rule)

    def test_zero_turnout(self):
        results, summary = self.vote([0.] * 10)
        self.assertEqual(results.votes.sum(), 0)
        self.assertEqual(results.vote_share.sum(), 0)
        self.assertEqual(summary["winner_ids"], [])
        self.assertTrue(summary["zero_turnout"])
        self.assertEqual(summary["abstentions"], 10)

    def test_full_turnout_and_argmax(self):
        results, summary = self.vote([1.] * 10, "argmax")
        self.assertEqual(results.votes.tolist(), [10, 0])
        self.assertEqual(summary["turnout"], 10)
        self.assertEqual(results.vote_share.sum(), 1.)

    def test_vote_accounting_and_fixed_seed(self):
        first, summary = self.vote([0.6] * 100)
        second, other = self.vote([0.6] * 100)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(summary, other)
        self.assertEqual(first.votes.sum(), summary["turnout"])
        self.assertEqual(summary["turnout"] + summary["abstentions"], 100)

    def test_ties_are_reported(self):
        results, summary = election_result(np.ones(2), np.eye(2), ["a", "b"],
                                           np.random.default_rng(1), np.random.default_rng(2))
        self.assertEqual(results.votes.tolist(), [1, 1])
        self.assertEqual(summary["winner_ids"], ["a", "b"])
        self.assertTrue(summary["tied"])


class SimulationTests(unittest.TestCase):
    def test_reproducibility_row_order_and_no_input_mutation(self):
        data = fixture()
        original = {key: value.copy(deep=True) for key, value in data.items()}
        config = SimulationConfig(days=5)
        first = simulate(config, data, 42)
        reordered = {key: value.iloc[::-1].reset_index(drop=True) for key, value in data.items()}
        second = simulate(config, reordered, 42)
        for name in ("daily_summary", "daily_support", "opinion_distribution", "candidate_results", "final_state", "sampled_state"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
        self.assertEqual(first.turnout, second.turnout)
        self.assertEqual(first.metadata["selected_table_sha256"], second.metadata["selected_table_sha256"])
        for key in original:
            pd.testing.assert_frame_equal(data[key], original[key])

    def test_frozen_control_and_daily_invariants(self):
        result = simulate(SimulationConfig(days=30, lambda_social=0), fixture(), 42)
        self.assertEqual(len(result.daily_summary), 31)
        np.testing.assert_array_equal(result.final_state.initial_opinion, result.final_state.final_opinion)
        np.testing.assert_allclose(result.daily_support.groupby("day").mean_support.sum(), 1.)
        self.assertTrue(result.opinion_distribution.groupby("day")["count"].sum().eq(3).all())
        self.assertTrue(result.daily_summary.polarization_variance.between(0, 1).all())

    def test_scaling_clipping_and_opinion_bounds(self):
        result = simulate(SimulationConfig(days=30, confidence_multiplier=10,
                                           susceptibility_multiplier=10), fixture(), 42)
        self.assertTrue(result.final_state.final_opinion.between(-1, 1).all())
        self.assertTrue(np.isfinite(result.daily_summary.select_dtypes("number")).all().all())

    def test_global_confidence_override(self):
        result = simulate(SimulationConfig(global_confidence=0), fixture(), 42)
        np.testing.assert_array_equal(result.final_state.initial_opinion, result.final_state.final_opinion)

    def test_rng_streams_keep_turnout_common_across_social_controls(self):
        data = fixture()
        frozen = simulate(SimulationConfig(lambda_social=0), data, 100)
        social = simulate(SimulationConfig(lambda_social=1), data, 100)
        self.assertEqual(frozen.turnout["turnout"], social.turnout["turnout"])
        self.assertEqual(frozen.metadata["child_seeds"], social.metadata["child_seeds"])
        streams = frozen.metadata["child_seeds"].values()
        self.assertEqual(len({tuple(value) for value in streams}), 4)

    def test_save_manifest_and_refuse_overwrite(self):
        result = simulate(SimulationConfig(days=1), fixture(), 42)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result.save(output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(manifest["config"]["days"], 1)
            self.assertEqual(len(list(output.glob("*.csv"))), 6)
            with self.assertRaises(FileExistsError):
                result.save(output)

    def test_sampled_view_matches_day_zero_and_final_state(self):
        result = simulate(SimulationConfig(days=5), fixture(), 42)
        self.assertEqual(len(result.sampled_state), 18)
        for day, column in ((0, "initial_opinion"), (5, "final_opinion")):
            sampled = result.sampled_state.loc[result.sampled_state.day.eq(day)]
            expected = result.final_state.set_index("voter_id").loc[sampled.voter_id, column]
            np.testing.assert_array_equal(sampled.opinion, expected)

    def test_invalid_config_and_seed(self):
        for kwargs in ({"days": 0}, {"days": True}, {"lambda_social": 2}, {"tau": 0},
                       {"rho": float("nan")}, {"global_confidence": 3}, {"vote_rule": "bad"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimulationConfig(**kwargs)
        with self.assertRaises(ValueError):
            simulate(SimulationConfig(), fixture(), -1)

    def test_mixed_constituency_and_nonfinite_inputs_rejected(self):
        for column, value in (("constituency_id", 2), ("initial_opinion", float("nan")),
                              ("turnout_propensity", 2.)):
            data = fixture()
            data["voters"].loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                simulate(SimulationConfig(), data, 42)


if __name__ == "__main__":
    unittest.main()
