"""Regional synchrony, controls, ballot isolation, campaigns and output accounting."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from election_sim import SimulationConfig, simulate, simulate_region, spillover_sweep
from election_sim.campaign.models import CampaignAction
from test_events import event_fixture, event


def region_fixture():
    a = event_fixture()
    b = {name: table.copy() for name, table in a.items()}
    b["voters"]["voter_id"] = ["d", "e", "f"]
    b["voters"]["constituency_id"] = 2
    b["voters"]["initial_opinion"] = [1., 0., -1.]
    b["candidates"]["candidate_id"] = ["left2", "right2"]
    b["candidates"]["constituency_id"] = 2
    b["social_network"]["source_voter_id"] = ["d", "e"]
    b["social_network"]["target_voter_id"] = ["e", "f"]
    data = {name: pd.concat([a[name], b[name]], ignore_index=True) for name in a}
    data["candidates"]["region_id"] = 1
    data["candidates"]["party_id"] = ["P1", "P2", "P1", "P2"]
    data["social_network"] = pd.concat([data["social_network"], pd.DataFrame([
        {"source_voter_id": "a", "target_voter_id": "e", "edge_weight": 1.},
        {"source_voter_id": "d", "target_voter_id": "b", "edge_weight": 1.}])], ignore_index=True)
    return data


def local_slice(data, cid):
    voters = data["voters"].loc[data["voters"].constituency_id == cid]
    ids = set(voters.voter_id)
    edges = data["social_network"]
    return {"voters": voters, "candidates": data["candidates"].loc[data["candidates"].constituency_id == cid],
            "social_network": edges.loc[edges.source_voter_id.isin(ids) & edges.target_voter_id.isin(ids)]}


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.data = region_fixture()
        self.config = SimulationConfig(days=3, lambda_social=.5, lambda_event=0)

    def test_zero_spillover_exactly_matches_standalone(self):
        region = simulate_region(self.config, [2, 1], 42, data=self.data, record_history=True)
        for cid in (1, 2):
            local = simulate(replace(self.config, constituency_id=cid), local_slice(self.data, cid), 42, record_history=True)
            pd.testing.assert_frame_equal(region.constituencies[cid].final_state, local.final_state)
            pd.testing.assert_frame_equal(region.constituencies[cid].candidate_results, local.candidate_results)
            np.testing.assert_array_equal(region.constituencies[cid].history["opinions"], local.history["opinions"])

    def test_cross_weights_and_synchronous_day_one(self):
        region = simulate_region(replace(self.config, days=1), [1, 2], 42, .2, data=self.data, record_history=True)
        state = region.constituencies[1].final_state
        # a=-1 and d=+1 influence b with raw weights 1 and .2.
        self.assertAlmostEqual(state.loc[1, "final_opinion"], .5 * (-1 + .2) / 1.2)
        # The d->e internal edge sees d's original +1; a->e sees original -1.
        self.assertAlmostEqual(region.constituencies[2].final_state.loc[1, "final_opinion"], .5*(1-.2)/1.2)
        self.assertEqual(region.metadata["cross_constituency_edge_count"], 2)

    def test_direction_confidence_and_no_cross_control(self):
        data = region_fixture()
        data["social_network"] = data["social_network"].iloc[:5]  # only a->e cross edge
        region = simulate_region(replace(self.config, days=1), [1, 2], spillover=1, data=data)
        control = simulate_region(replace(self.config, days=1), [1, 2], data=data)
        pd.testing.assert_frame_equal(region.constituencies[1].final_state, control.constituencies[1].final_state)
        self.assertNotEqual(region.constituencies[2].final_state.loc[1, "final_opinion"], control.constituencies[2].final_state.loc[1, "final_opinion"])
        data["voters"].loc[data["voters"].voter_id.eq("e"), "confidence_threshold"] = 0
        frozen = simulate_region(replace(self.config, days=1), [1, 2], spillover=1, data=data)
        self.assertEqual(frozen.constituencies[2].final_state.loc[1, "final_opinion"], 0)

    def test_order_independence_and_history_alignment(self):
        a = simulate_region(self.config, [1, 2], spillover=.1, data=self.data, record_history=True)
        shuffled = {name: table.sample(frac=1, random_state=9) for name, table in self.data.items()}
        b = simulate_region(self.config, [2, 1], spillover=.1, data=shuffled, record_history=True)
        pd.testing.assert_frame_equal(a.candidate_results, b.candidate_results)
        np.testing.assert_array_equal(a.history["opinions"], b.history["opinions"])
        for cid in (1, 2):
            index = np.flatnonzero(a.history["constituency_ids"] == cid)
            np.testing.assert_array_equal(a.history["opinions"][:, index], a.constituencies[cid].history["opinions"])

    def test_party_campaigns_agents_and_local_poll_ballots(self):
        result = simulate_region(replace(self.config, lambda_campaign=1), [1, 2], data=self.data,
            party_campaigns=[dict(action_id="party", party_id="P1", start_day=1, duration=1,
                                 intensity=.5, reach_fraction=1, message_type="smear", target_party_id="P2")],
            party_agents=[dict(party_id="P2", strategy="greedy", budget=3, decide_every=1, reach_fraction=1)])
        for cid, local in result.constituencies.items():
            candidate_ids = set(self.data["candidates"].loc[self.data["candidates"].constituency_id == cid].candidate_id)
            self.assertEqual(set(local.diagnostics["polls"][0]["shares"]), candidate_ids)
            self.assertTrue(all(value >= 0 for value in local.diagnostics["agent_budget_left"].values()))
            self.assertEqual(local.diagnostics["campaign_audiences"][0]["selected_recipients"], 3)

    def test_event_geography(self):
        data = {**self.data, "events": event(event_day=1, constituency_id=1, candidate_id="left")}
        config = replace(self.config, lambda_event=1)
        result = simulate_region(config, [1, 2], data=data)
        self.assertEqual(result.constituencies[1].diagnostics["events_loaded"], 1)
        self.assertEqual(result.constituencies[2].diagnostics["events_loaded"], 0)

    def test_accounting_save_and_sweep(self):
        data = region_fixture()
        data["voters"]["turnout_propensity"] = 1
        result = simulate_region(self.config, [1, 2], data=data, record_history=True)
        self.assertEqual(result.candidate_results.votes.sum(), 6)
        self.assertEqual(result.turnout.turnout.sum(), 6)
        self.assertEqual(result.seats.seats_won.sum() + len(result.metadata["unresolved_constituencies"]), 2)
        with TemporaryDirectory() as folder:
            result.save(Path(folder) / "run")
            self.assertTrue((Path(folder) / "run/C02/history.npz").exists())
            self.assertTrue((Path(folder) / "run/region_history.npz").exists())
        sweep = spillover_sweep(self.config, [1, 2], data=data)
        self.assertEqual(len(sweep), 16)
        self.assertEqual(set(sweep.spillover), {0, .05, .1, .2})

    def test_invalid_inputs_and_ballot_crossing(self):
        for value in (-1, 2, float("nan"), True):
            with self.assertRaises(ValueError):
                simulate_region(self.config, [1, 2], spillover=value, data=self.data)
        for ids in ([], [1, 1], [1, 3]):
            with self.assertRaises(ValueError):
                simulate_region(self.config, ids, data=self.data)
        bad = region_fixture()
        bad["voters"].loc[bad["voters"].constituency_id.eq(2), "region_id"] = 2
        with self.assertRaises(ValueError):
            simulate_region(self.config, [1, 2], data=bad)
        with self.assertRaises(ValueError):
            simulate_region(replace(self.config, lambda_campaign=1), [1, 2], data=self.data,
                campaign_actions=[CampaignAction("bad", "left", 1, 1, 1, 1, "smear", target_candidate_id="right2")])
        with self.assertRaises(ValueError):
            simulate_region(self.config, [1, 2], data=self.data, party_agents=[dict(party_id="missing")])

    def test_zero_control_with_agents_news_and_mobilization(self):
        from election_sim import CandidateAgent
        config = replace(self.config, lambda_campaign=1, lambda_event=1, gamma_mob=.1, gamma_event=.2)
        data = {**self.data, "events": event(candidate_id=None, candidate_effect=None,
            constituency_id=None, region_id=1, event_day=1)}
        result = simulate_region(config, [1, 2], 22, data=data,
            party_agents=[dict(party_id="P1", budget=3, reach_fraction=1, decide_every=1)])
        for cid, candidate in ((1, "left"), (2, "left2")):
            local = {**local_slice(data, cid), "events": data["events"]}
            control = simulate(replace(config, constituency_id=cid), local, 22,
                agents=[CandidateAgent(candidate, budget=3, reach_fraction=1, decide_every=1)])
            pd.testing.assert_frame_equal(result.constituencies[cid].final_state, control.final_state)
            self.assertEqual(result.constituencies[cid].turnout, control.turnout)

    def test_ties_and_zero_turnout_do_not_award_seats(self):
        data = region_fixture()
        data["voters"]["turnout_propensity"] = 0
        result = simulate_region(self.config, [1, 2], data=data)
        self.assertEqual(result.seats.seats_won.sum(), 0)
        self.assertEqual(result.metadata["unresolved_constituencies"], [1, 2])
        # Three voters choose opposite candidates deterministically in each seat;
        # abstaining the middle voter gives a 1-1 tie on each ballot.
        data["voters"]["turnout_propensity"] = [1, 0, 1, 1, 0, 1]
        tied = simulate_region(replace(self.config, lambda_social=0, vote_rule="argmax"), [1, 2], data=data)
        self.assertEqual(tied.seats.seats_won.sum(), 0)
        self.assertEqual(tied.seats.tied_constituencies.tolist(), [2, 2])
