"""Step 3 integration and probability checks."""
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from election_sim import NewsAgent, SimulationConfig, simulate
from election_sim.campaign.models import CampaignAction
from election_sim.voting import mobilized_propensity
from test_events import event_fixture, event


class NewsTests(unittest.TestCase):
    def setUp(self):
        self.data = event_fixture()
        self.config = SimulationConfig(days=3, lambda_social=0, lambda_campaign=1,
                                       lambda_event=1, poll_noise=0)
        self.smear = CampaignAction("smear", "right", 1, 1, 1, 1,
                                   "smear", target_candidate_id="left")

    def run_news(self, rows, **kwargs):
        return simulate(self.config, self.data, 17, [self.smear], record_history=True,
                        news_agent=NewsAgent(rows), **kwargs)

    def test_fact_check_clears_belief_and_penalty_on_correct_day(self):
        result = self.run_news(event(event_type="fact_check", event_day=2, candidate_effect=0))
        self.assertTrue((result.history["smear_belief"][1, :, 0] == 1).all())
        self.assertTrue((result.history["smear_belief"][2, :, 0] == 2).all())
        self.assertTrue((result.history["support"][2, :, 0] > result.history["support"][1, :, 0]).all())
        self.assertEqual(result.diagnostics["event_applications"][0]["accepted"], 3)

    def test_fact_check_zero_credibility_and_group_scope(self):
        zero = self.run_news(event(event_type="fact_check", event_day=2, credibility=0))
        self.assertTrue((zero.history["smear_belief"][2, :, 0] == 1).all())
        group = self.run_news(event(event_type="fact_check", event_day=2, target_group="Students"))
        np.testing.assert_array_equal(group.history["smear_belief"][2, :, 0], [1, 2, 1])

    def test_public_poll_is_current_and_does_not_move_support(self):
        result = self.run_news(event(event_type="poll_release", event_day=2, candidate_id=None,
                                     candidate_effect=None))
        poll = result.diagnostics["public_polls"][0]
        self.assertEqual(poll["day"], 2)
        self.assertEqual(poll["announcement"], "public")
        leaders = result.history["support"][2].argmax(axis=1)
        self.assertAlmostEqual(poll["shares"]["left"], (leaders == 0).mean())
        control = self.run_news(pd.DataFrame())
        np.testing.assert_array_equal(result.history["support"], control.history["support"])

    def test_seeded_generation_and_csv(self):
        agent = NewsAgent(random_events_per_day=3)
        a = simulate(self.config, self.data, 17, news_agent=agent)
        b = simulate(self.config, self.data, 17, news_agent=agent)
        pd.testing.assert_frame_equal(a.final_state, b.final_state)
        self.assertGreater(a.diagnostics["events_loaded"], 0)
        with TemporaryDirectory() as folder:
            a.save(Path(folder) / "generated")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "schedule.csv"
            event(event_type="fact_check", event_day=2).to_csv(path, index=False)
            result = simulate(self.config, self.data, 17, [self.smear], news_agent=NewsAgent(path))
            self.assertEqual(result.diagnostics["smear_rejecters"]["left"], 3)

    def test_zero_strength_disables_news_and_exposure(self):
        data = {**self.data, "events": event(event_type="fact_check", event_day=2)}
        result = simulate(replace(self.config, lambda_event=0), data, 17, [self.smear])
        self.assertEqual(result.final_state.event_exposure.sum(), 0)
        self.assertEqual(result.diagnostics["smear_rejecters"]["left"], 0)

    def test_unique_daily_exposure_and_turnout_integration(self):
        data = self.data
        data["voters"]["turnout_propensity"] = .5
        second = replace(self.smear, action_id="second")
        result = simulate(replace(self.config, gamma_mob=1), data, 17, [self.smear, second])
        np.testing.assert_array_equal(result.final_state.campaign_exposure, 1)
        np.testing.assert_allclose(result.final_state.turnout_probability, 1/(1+np.exp(-1)))
        news = simulate(replace(self.config, gamma_event=-1),
            {**data, "events": event(duration=2)}, 17)
        np.testing.assert_allclose(news.final_state.turnout_probability, 1/(1+np.exp(2)))

    def test_turnout_defaults_endpoints_and_extreme_coefficients(self):
        p = np.array([0., .5, 1.])
        np.testing.assert_array_equal(mobilized_propensity(p, [2]*3, [4]*3), p)
        positive = mobilized_propensity(p, [1]*3, [0]*3, 1000)
        negative = mobilized_propensity(p, [1]*3, [0]*3, -1000)
        np.testing.assert_array_equal(positive, 1)
        np.testing.assert_array_equal(negative, 0)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SimulationConfig(gamma_mob=value)
        with self.assertRaises(ValueError):
            NewsAgent(random_events_per_day=-1)

    def test_invalid_announcement_and_history_round_trip(self):
        with self.assertRaises(ValueError):
            self.run_news(event(event_type="poll_release", target_group="Students"))
        with self.assertRaises(ValueError):
            self.run_news(event(event_type="fact_check", candidate_id=None))
        result = self.run_news(event(event_type="fact_check", event_day=2))
        self.assertIn("fact_check", [row["kind"] for row in result.history["timeline"]])
        with TemporaryDirectory() as folder:
            result.save(Path(folder) / "run")
            self.assertTrue((Path(folder) / "run" / "manifest.json").exists())
