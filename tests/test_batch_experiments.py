"""General matched-seed batch and campaign comparison tests."""

import unittest

from election_sim import SimulationConfig
from election_sim.campaign import CampaignAction
from election_sim.experiments import (build_step5_scenarios, compare_campaign,
                                     run_batch, summarize_batch)
from test_baseline import fixture
from test_region import region_fixture


class BatchExperimentTests(unittest.TestCase):
    def test_batch_is_tidy_reproducible_and_summarizable(self):
        data = fixture()
        config = SimulationConfig(days=2, lambda_social=0, lambda_event=0)
        scenarios = {
            "frozen": {"config_overrides": {"lambda_social": 0}},
            "social": {"config_overrides": {"lambda_social": 0.2}},
        }
        first = run_batch(config, data, scenarios, [4, 8])
        second = run_batch(config, data, scenarios, [4, 8])
        self.assertEqual(len(first), 2 * 2 * 2)
        self.assertEqual(set(first.columns) & {
            "scenario", "seed", "constituency_id", "candidate_id", "vote_share",
            "mean_support", "turnout_rate", "winner"}, {
            "scenario", "seed", "constituency_id", "candidate_id", "vote_share",
            "mean_support", "turnout_rate", "winner"})
        self.assertEqual(first.to_dict("records"), second.to_dict("records"))
        summary = summarize_batch(first)
        self.assertEqual(len(summary), 2 * 2)
        self.assertEqual(summary.seed_count.unique().tolist(), [2])
        self.assertTrue(summary.mean_support_p05.le(summary.mean_support_median).all())
        self.assertTrue(summary.mean_support_median.le(summary.mean_support_p95).all())

    def test_campaign_comparison_pairs_same_seed_candidates(self):
        data = fixture()
        config = SimulationConfig(days=2, lambda_social=0, lambda_campaign=1,
                                  lambda_event=0, vote_rule="argmax")
        action = CampaignAction("ordinary", "left", 1, 2, 1, 1)
        result = compare_campaign(config, data, [3, 7], treatment_actions=[action])
        self.assertEqual(len(result), 2 * 2)
        self.assertEqual(result.seed.nunique(), 2)
        self.assertEqual(set(result.columns) & {
            "mean_support_difference", "vote_share_difference",
            "turnout_rate_difference", "candidate_winner_status_changed",
            "constituency_winner_changed"}, {
            "mean_support_difference", "vote_share_difference",
            "turnout_rate_difference", "candidate_winner_status_changed",
            "constituency_winner_changed"})
        self.assertTrue(result.mean_support_difference.notna().all())

    def test_batch_rejects_bad_scenarios_and_duplicate_seeds(self):
        config = SimulationConfig(days=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            run_batch(config, fixture(), {"base": {}}, [2, 2])
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            run_batch(config, fixture(), {"base": {"unknown": True}}, [2])
        with self.assertRaisesRegex(ValueError, "require constituency_ids"):
            run_batch(config, fixture(), {"base": {"spillover": 0.1}}, [2])
        with self.assertRaisesRegex(ValueError, "config_overrides must be a mapping"):
            run_batch(config, fixture(), {"base": {"config_overrides": []}}, [2])

    def test_step5_matrix_covers_requested_experiment_arms(self):
        data = fixture()
        scenarios = build_step5_scenarios(
            SimulationConfig(days=2, lambda_campaign=0.2), data)
        self.assertTrue({"frozen", "social_only", "campaign_ordinary",
                         "campaign_misleading", "campaign_smear", "social_plus_news",
                         "all_together", "scripted_equal_budget", "agents_greedy",
                         "agents_go_negative", "agents_rebut", "target_uniform",
                         "target_high_susceptibility", "target_high_influence",
                         "target_community"}.issubset(scenarios))
        actions = scenarios["campaign_misleading"]["campaign_actions"]
        self.assertEqual(actions[0].message_type, "misleading")
        self.assertEqual(actions[0].distortion, 0.5)

    def test_batch_emits_constituency_rows_for_regional_scenarios(self):
        data = region_fixture()
        config = SimulationConfig(days=1, lambda_social=0.2, lambda_event=0)
        result = run_batch(config, data,
                           {"regional": {"constituency_ids": [1, 2],
                                          "spillover": 0.1}}, [12])
        self.assertEqual(set(result.constituency_id), {1, 2})
        self.assertEqual(result.groupby("constituency_id").candidate_id.nunique().to_dict(),
                         {1: 2, 2: 2})
        self.assertTrue(result.mean_support.notna().all())


if __name__ == "__main__":
    unittest.main()
