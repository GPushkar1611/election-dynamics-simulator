"""Full per-day history: shapes, consistency with other outputs, save/load.

Run from the repository root:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from election_sim import SimulationConfig, load_history, simulate
from election_sim.campaign import CampaignAction
from test_baseline import fixture
from test_events import event, event_fixture

DAYS = 4
ORDINARY = CampaignAction("ordinary", "left", 2, 2, 0.5, 2 / 3)
SMEAR = CampaignAction("smear", "left", 3, 1, 0.5, 1 / 3,
                       message_type="smear", target_candidate_id="right")


def run(record_history=True, actions=(ORDINARY, SMEAR)):
    config = SimulationConfig(days=DAYS, lambda_social=0.2, lambda_campaign=0.5)
    return simulate(config, fixture(), 42, list(actions), record_history=record_history)


class HistoryTests(unittest.TestCase):
    def test_off_by_default(self):
        result = simulate(SimulationConfig(days=DAYS), fixture(), 42)
        self.assertIsNone(result.history)
        self.assertFalse(result.metadata["history_recorded"])

    def test_shapes_and_ids(self):
        history = run().history
        self.assertEqual(history["opinions"].shape, (DAYS + 1, 3))
        self.assertEqual(history["support"].shape, (DAYS + 1, 3, 2))
        self.assertEqual(history["campaign_exposed"].shape, (DAYS + 1, 3, 2))
        self.assertEqual(history["event_exposed"].shape, (DAYS + 1, 3))
        self.assertEqual(history["smear_belief"].shape, (DAYS + 1, 3, 2))
        self.assertEqual(history["voter_ids"].tolist(), ["a", "b", "c"])
        self.assertEqual(history["candidate_ids"].tolist(), ["left", "right"])
        self.assertEqual(history["action_ids"].tolist(), ["ordinary", "smear"])

    def test_matches_initial_final_and_sampled_outputs(self):
        result = run()
        history = result.history
        np.testing.assert_allclose(history["opinions"][0], result.final_state.initial_opinion, atol=1e-6)
        np.testing.assert_allclose(history["opinions"][-1], result.final_state.final_opinion, atol=1e-6)
        for column, candidate in enumerate(history["candidate_ids"]):
            np.testing.assert_allclose(history["support"][-1, :, column],
                                       result.final_state[f"support_{candidate}"], atol=1e-6)
        daily_mean = result.daily_support.pivot(index="day", columns="candidate_id",
                                                values="mean_support")
        np.testing.assert_allclose(history["support"].mean(axis=1), daily_mean.to_numpy(), atol=1e-6)

    def test_recording_does_not_change_results(self):
        with_history, without = run(True), run(False)
        np.testing.assert_array_equal(with_history.final_state.final_opinion,
                                      without.final_state.final_opinion)
        self.assertTrue(with_history.candidate_results.equals(without.candidate_results))

    def test_campaign_exposure_only_on_active_days(self):
        exposed = run().history["campaign_exposed"]
        per_day = exposed.sum(axis=1)  # [day, action] recipient counts
        np.testing.assert_array_equal(per_day[:, 0], [0, 0, 2, 2, 0])  # ordinary: days 2-3
        np.testing.assert_array_equal(per_day[:, 1], [0, 0, 0, 1, 0])  # smear: day 3

    def test_event_exposure_and_timeline(self):
        data = {**event_fixture(), "events": event(event_day=2, duration=2)}
        history = simulate(SimulationConfig(days=DAYS, lambda_social=0), data, 17,
                           record_history=True).history
        np.testing.assert_array_equal(history["event_exposed"].sum(axis=1), [0, 0, 3, 3, 0])
        self.assertEqual(history["timeline"], [{
            "day": 2, "end_day": 3, "kind": "event", "id": "event-1",
            "label": "Policy announcement", "candidate_id": "left"}])

    def test_zero_event_strength_records_no_event_exposure(self):
        data = {**event_fixture(), "events": event(event_day=2)}
        history = simulate(SimulationConfig(days=DAYS, lambda_social=0, lambda_event=0), data, 17,
                           record_history=True).history
        self.assertFalse(history["event_exposed"].any())
        self.assertEqual(history["timeline"], [])

    def test_smear_belief_recorded_from_smear_day(self):
        belief = run().history["smear_belief"][:, :, 1]  # claims about "right"
        np.testing.assert_array_equal((belief == 1).sum(axis=1), [0, 0, 0, 1, 1])

    def test_campaign_timeline(self):
        timeline = run().history["timeline"]
        self.assertEqual([(item["day"], item["end_day"], item["kind"]) for item in timeline],
                         [(2, 3, "campaign_ordinary"), (3, 3, "campaign_smear")])
        self.assertEqual(timeline[1]["label"], "smear by left -> right")

    def test_save_and_load_round_trip(self):
        result = run()
        with TemporaryDirectory() as folder:
            result.save(Path(folder) / "run")
            loaded = load_history(Path(folder) / "run")
        self.assertEqual(loaded["timeline"], result.history["timeline"])
        for name in ("voter_ids", "candidate_ids", "action_ids", "opinions", "support",
                     "smear_belief", "campaign_exposed", "event_exposed"):
            np.testing.assert_array_equal(loaded[name], result.history[name])

    def test_no_history_files_when_off(self):
        with TemporaryDirectory() as folder:
            run(False).save(Path(folder) / "run")
            self.assertFalse((Path(folder) / "run" / "history.npz").exists())


if __name__ == "__main__":
    unittest.main()
