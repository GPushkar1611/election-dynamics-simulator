"""Matched scalar versus synthetic issue-aware support experiments."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from election_sim import CandidateAgent, SimulationConfig, run_support_sensitivity_sweep, simulate
from election_sim.support import ISSUES
from test_baseline import fixture


def issue_fixture():
    data = fixture()
    for issue in ISSUES:
        data["voters"][f"issue_{issue}"] = 0.2
    # Give candidates different cross-issue profiles so weighted fit is testable.
    data["candidates"].loc[0, [f"position_{issue}" for issue in ISSUES]] = [1, -1, 1, -1, 0]
    data["candidates"].loc[1, [f"position_{issue}" for issue in ISSUES]] = [-1, 1, -1, 1, 0]
    return data


class SupportSensitivityTests(unittest.TestCase):
    def test_matched_sweep_has_scalar_and_each_profile_dispersion(self):
        data = issue_fixture()
        result = run_support_sensitivity_sweep(
            SimulationConfig(days=2, lambda_social=0), data, [4, 8], [0.0, 0.5])
        self.assertEqual(len(result.per_seed), 2 * 3)
        self.assertEqual(len(result.per_seed_candidate), 2 * 3 * 2)
        self.assertEqual(set(result.per_seed.support_mode),
                         {"scalar", "synthetic_issue_profile"})
        self.assertEqual(result.summary.seed_count.unique().tolist(), [2])
        scalar = result.per_seed.query("support_mode == 'scalar'")
        self.assertTrue(scalar.dispersion.isna().all())
        self.assertEqual(result.metadata["seeds"], [4, 8])

    def test_seed_matching_preserves_turnout_and_nonprofile_streams(self):
        data = issue_fixture()
        scalar = simulate(SimulationConfig(days=2, lambda_social=0), data, 13)
        issue_aware = simulate(SimulationConfig(
            days=2, lambda_social=0, support_mode="synthetic_issue_profile"), data, 13)
        self.assertEqual(scalar.turnout["turnout"], issue_aware.turnout["turnout"])
        for stream in ("campaign", "turnout", "voting", "community", "events", "polls", "spread"):
            self.assertEqual(scalar.metadata["child_seeds"][stream],
                             issue_aware.metadata["child_seeds"][stream])

    def test_agent_sweep_records_reproducible_decisions_for_each_arm(self):
        data = issue_fixture()
        config = SimulationConfig(days=2, lambda_social=0, lambda_campaign=0.5,
                                  lambda_event=0, poll_sample_size=3, poll_noise=0)
        agents = [CandidateAgent("left", start_day=1, decide_every=1,
                                 reach_fraction=0.34, budget=100),
                  CandidateAgent("right", start_day=1, decide_every=1,
                                 reach_fraction=0.34, budget=100)]
        first = run_support_sensitivity_sweep(config, data, [3], [0.5], agents=agents)
        second = run_support_sensitivity_sweep(config, data, [3], [0.5], agents=agents)
        self.assertEqual(len(first.per_seed_agent_decisions), 8)
        self.assertEqual(first.per_seed.agent_decision_count.tolist(), [4, 4])
        self.assertEqual(set(first.per_seed_agent_decisions.support_mode),
                         {"scalar", "synthetic_issue_profile"})
        scalar = first.per_seed.query("support_mode == 'scalar'").iloc[0]
        self.assertEqual(scalar.agent_actions_changed_vs_scalar, 0)
        self.assertEqual(scalar.mean_absolute_agent_poll_share_difference_vs_scalar, 0)
        self.assertIn("mean_agent_action_change_rate_vs_scalar", first.summary.columns)
        pd.testing.assert_frame_equal(first.per_seed_agent_decisions,
                                      second.per_seed_agent_decisions)
        self.assertEqual(len(first.metadata["agents"]), len(agents))

    def test_sweep_validates_seed_dispersion_and_weights(self):
        data = issue_fixture()
        config = SimulationConfig(days=1)
        with self.assertRaisesRegex(ValueError, "at least one seed"):
            run_support_sensitivity_sweep(config, data, [], [0.5])
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            run_support_sensitivity_sweep(config, data, [1], [1.1])
        missing = {**data, "voters": data["voters"].drop(columns="issue_economy")}
        with self.assertRaisesRegex(ValueError, "needs voter issue weights"):
            run_support_sensitivity_sweep(config, missing, [1], [0.5])

    def test_sweep_saves_manifest_and_all_tables(self):
        result = run_support_sensitivity_sweep(
            SimulationConfig(days=1, lambda_social=0), issue_fixture(), [6], [0.0])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "sensitivity"
            result.save(output)
            for name in ("per_seed.csv", "per_seed_candidate.csv",
                         "per_seed_agent_decisions.csv", "summary.csv",
                         "candidate_summary.csv", "manifest.json"):
                self.assertTrue((output / name).exists())
            self.assertEqual(len(pd.read_csv(output / "summary.csv")), 2)


if __name__ == "__main__":
    unittest.main()
