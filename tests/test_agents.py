"""Polls and candidate agents: what they see, what they decide, budgets.

Run from the repository root:
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from election_sim import CandidateAgent, SimulationConfig, load_agents, simulate
from election_sim.agents import check_agents, run_poll
from election_sim.support import ISSUES


def crowd(size=40):
    """`size` voters spread from -1 to 1 on a ring, alternating Student/Farmer
    and Urban/Rural; two candidates at -1 ("left") and +1 ("right")."""
    ids = [f"v{i:02d}" for i in range(size)]
    voters = pd.DataFrame({
        "voter_id": ids, "constituency_id": 1,
        "initial_opinion": np.linspace(-1, 1, size), "susceptibility": 1.0,
        "confidence_threshold": 2.0, "influence_power": 1.0, "turnout_propensity": 1.0,
        "occupation": ["Student", "Farmer"] * (size // 2),
        "urban_rural": ["Urban", "Rural"] * (size // 2),
        "wealth_level": "Low",
    })
    candidates = pd.DataFrame({
        "candidate_id": ["left", "right"], "constituency_id": [1, 1], "local_popularity": [0.5, 0.5],
        **{f"position_{issue}": [-1.0, 1.0] for issue in ISSUES},
    })
    edges = pd.DataFrame({"source_voter_id": ids, "target_voter_id": ids[1:] + ids[:1],
                          "edge_weight": 1.0})
    return {"voters": voters, "candidates": candidates, "social_network": edges}


def poll(shares, groups=None):
    return {"day": 1, "sample_size": 100, "shares": shares, "group_shares": groups or {}}


CONFIG = SimulationConfig(days=6, lambda_social=0, lambda_campaign=1,
                          poll_sample_size=40, poll_noise=0)


class PollTests(unittest.TestCase):
    def setUp(self):
        self.voters = crowd()["voters"]
        # Voters 0-9 lean left, 10-39 lean right.
        self.support = np.tile([0.2, 0.8], (40, 1))
        self.support[:10] = [0.9, 0.1]

    def test_full_noiseless_poll_matches_true_leanings(self):
        result = run_poll(self.support, self.voters, ["left", "right"], np.random.default_rng(0), 1, 40)
        self.assertEqual(result["shares"], {"left": 0.25, "right": 0.75})
        # 20 students (even indices): voters 0,2,..,8 lean left -> 5/20.
        self.assertEqual(result["group_shares"]["Students"], {"left": 0.25, "right": 0.75})

    def test_small_groups_are_not_reported(self):
        result = run_poll(self.support, self.voters, ["left", "right"], np.random.default_rng(0), 1, 40)
        self.assertNotIn("Unemployed", result["group_shares"])  # nobody in it
        self.assertNotIn("General", result["group_shares"])     # that is just "shares"
        self.assertEqual(result["group_shares"]["Low-income"], result["shares"])

    def test_noisy_poll_is_a_reproducible_distribution(self):
        first = run_poll(self.support, self.voters, ["left", "right"], np.random.default_rng(3), 1, 20, 0.1)
        second = run_poll(self.support, self.voters, ["left", "right"], np.random.default_rng(3), 1, 20, 0.1)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(first["shares"].values()), 1.0)
        self.assertTrue(all(value >= 0 for value in first["shares"].values()))
        self.assertEqual(first["sample_size"], 20)


class DecisionTests(unittest.TestCase):
    groups = {"Students": {"left": 0.6, "right": 0.4}, "Farmers": {"left": 0.2, "right": 0.8}}

    def test_greedy_targets_the_weakest_group(self):
        action = CandidateAgent("left").decide(4, 30, poll({"left": 0.4, "right": 0.6}, self.groups))
        self.assertEqual((action.message_type, action.target_policy, action.target_group),
                         ("ordinary", "group", "Farmers"))
        self.assertEqual((action.action_id, action.start_day, action.duration),
                         ("left_day04_ordinary", 4, 3))

    def test_greedy_without_group_data_goes_uniform(self):
        action = CandidateAgent("left").decide(1, 30, poll({"left": 0.4, "right": 0.6}))
        self.assertEqual(action.target_policy, "uniform")

    def test_go_negative_smears_the_leader_only_when_far_behind(self):
        agent = CandidateAgent("left", strategy="go_negative", smear_gap=0.1, policy="high_influence")
        far = agent.decide(1, 30, poll({"left": 0.3, "right": 0.7}, self.groups))
        self.assertEqual((far.message_type, far.target_candidate_id, far.target_policy),
                         ("smear", "right", "high_influence"))
        close = agent.decide(1, 30, poll({"left": 0.46, "right": 0.54}, self.groups))
        self.assertEqual(close.message_type, "ordinary")
        leading = agent.decide(1, 30, poll({"left": 0.9, "right": 0.1}, self.groups))
        self.assertEqual(leading.message_type, "ordinary")

    def test_rebut_reacts_to_a_drop_with_a_broad_message(self):
        agent = CandidateAgent("left", strategy="rebut", drop_alert=0.02)
        before = poll({"left": 0.6, "right": 0.4}, self.groups)
        dropped = agent.decide(4, 30, poll({"left": 0.5, "right": 0.5}, self.groups), before)
        self.assertEqual(dropped.target_policy, "uniform")
        steady = agent.decide(4, 30, poll({"left": 0.6, "right": 0.4}, self.groups), before)
        self.assertEqual(steady.target_policy, "group")
        first = agent.decide(1, 30, poll({"left": 0.5, "right": 0.5}, self.groups), None)
        self.assertEqual(first.target_policy, "group")

    def test_distortion_makes_own_messages_misleading(self):
        action = CandidateAgent("left", distortion=0.4).decide(1, 30, poll({"left": 0.5, "right": 0.5}))
        self.assertEqual((action.message_type, action.distortion), ("misleading", 0.4))

    def test_schedule_and_last_action_is_cut_at_election_day(self):
        agent = CandidateAgent("left", start_day=10, decide_every=3)
        self.assertEqual([day for day in range(1, 31) if agent.decides_on(day)],
                         [10, 13, 16, 19, 22, 25, 28])
        self.assertEqual(agent.decide(28, 30, poll({"left": 0.5, "right": 0.5})).duration, 3)
        self.assertEqual(agent.decide(29, 30, poll({"left": 0.5, "right": 0.5})).duration, 2)

    def test_invalid_agents_are_rejected(self):
        for kwargs, message in (({"strategy": "bribe"}, "strategy"), ({"policy": "group"}, "policy"),
                                ({"budget": -1}, "budget"), ({"decide_every": 0}, "decide_every"),
                                ({"reach_fraction": 2}, "reach_fraction")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                CandidateAgent("left", **kwargs)
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            check_agents([CandidateAgent("nobody")], {"left", "right"})
        with self.assertRaisesRegex(ValueError, "more than one"):
            check_agents([CandidateAgent("left"), CandidateAgent("left")], {"left", "right"})

    def test_load_agents_from_json(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agents.json"
            path.write_text(json.dumps({"agents": [{"candidate_id": "right", "strategy": "rebut"},
                                                   {"candidate_id": "left"}]}), encoding="utf-8")
            agents = load_agents(path, {"left", "right"})
        self.assertEqual([agent.candidate_id for agent in agents], ["left", "right"])
        self.assertEqual(agents[1].strategy, "rebut")


class AgentSimulationTests(unittest.TestCase):
    def test_agents_poll_decide_and_act_on_their_days(self):
        agents = [CandidateAgent("left", reach_fraction=0.25, decide_every=2),
                  CandidateAgent("right", start_day=3, reach_fraction=0.25, decide_every=2)]
        result = simulate(CONFIG, crowd(), 42, agents=agents)
        decisions = pd.DataFrame(result.diagnostics["agent_decisions"])
        self.assertEqual(decisions[["day", "candidate_id"]].values.tolist(),
                         [[1, "left"], [3, "left"], [3, "right"], [5, "left"], [5, "right"]])
        self.assertTrue(decisions.taken.all())
        self.assertEqual([poll["day"] for poll in result.diagnostics["polls"]], [1, 3, 5])
        self.assertEqual(result.diagnostics["campaign_actions_loaded"], 5)
        self.assertEqual(result.metadata["scenario"], "agent_campaign")
        # Each action reaches 10 voters for 2 days (the day-5 ones are cut at day 6).
        self.assertEqual(result.diagnostics["campaign_exposures"], 5 * 10 * 2)

    def test_budget_is_never_exceeded(self):
        agent = CandidateAgent("left", reach_fraction=0.25, decide_every=2, budget=45)
        result = simulate(CONFIG, crowd(), 42, agents=[agent])
        decisions = pd.DataFrame(result.diagnostics["agent_decisions"])
        self.assertEqual(decisions.taken.tolist(), [True, True, False])  # costs 20, 20, 20 > 5 left
        self.assertEqual(result.diagnostics["agent_budget_left"], {"left": 5})
        self.assertEqual(decisions.loc[decisions.taken, "cost"].sum(), 40)

    def test_zero_budget_agent_changes_nothing(self):
        control = simulate(CONFIG, crowd(), 42)
        idle = simulate(CONFIG, crowd(), 42, agents=[CandidateAgent("left", budget=0)])
        pd.testing.assert_frame_equal(control.final_state, idle.final_state)
        pd.testing.assert_frame_equal(control.candidate_results, idle.candidate_results)

    def test_campaigning_agent_gains_support_and_runs_reproduce(self):
        agent = CandidateAgent("left", reach_fraction=0.5, intensity=1.0)
        control = simulate(CONFIG, crowd(), 42)
        first = simulate(CONFIG, crowd(), 42, agents=[agent])
        second = simulate(CONFIG, crowd(), 42, agents=[agent])
        pd.testing.assert_frame_equal(first.final_state, second.final_state)
        self.assertEqual(first.diagnostics["agent_decisions"], second.diagnostics["agent_decisions"])
        self.assertGreater(first.final_state.support_left.mean(), control.final_state.support_left.mean())

    def test_trailing_go_negative_agent_smears_the_leader(self):
        data = crowd()
        data["candidates"].loc[1, "local_popularity"] = 0.9  # "right" leads clearly
        agent = CandidateAgent("left", strategy="go_negative", reach_fraction=0.25, decide_every=6)
        result = simulate(CONFIG, data, 42, agents=[agent])
        decision, = result.diagnostics["agent_decisions"]
        self.assertEqual((decision["message_type"], decision["target_candidate_id"]), ("smear", "right"))
        self.assertEqual(result.diagnostics["smear_believers"]["right"], 10)

    def test_agent_actions_appear_in_history(self):
        result = simulate(CONFIG, crowd(), 42, agents=[CandidateAgent("left", decide_every=3)],
                          record_history=True)
        history = result.history
        self.assertEqual(history["action_ids"].tolist(), ["left_day01_ordinary", "left_day04_ordinary"])
        self.assertEqual(history["campaign_exposed"].sum(axis=1)[:, 0].tolist(), [0, 2, 2, 2, 0, 0, 0])
        self.assertEqual([item["day"] for item in history["timeline"]], [1, 4])

    def test_agents_need_campaign_strength(self):
        with self.assertRaisesRegex(ValueError, "lambda_campaign is 0"):
            simulate(SimulationConfig(days=2), crowd(), 42, agents=[CandidateAgent("left")])


if __name__ == "__main__":
    unittest.main()
