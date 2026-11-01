"""Small integration checks for the supplied constituency data."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

try:
    from .loader import load_constituency
except ImportError:
    from loader import load_constituency


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic_election_dataset"


class LoaderTests(unittest.TestCase):
    def test_c01_is_a_clean_local_slice(self):
        result = load_constituency(DATA, 1)
        voter_ids = set(result["voters"].voter_id)
        edges = result["social_network"]
        self.assertEqual(result["diagnostics"]["constituency_id"], 1)
        self.assertEqual(result["voters"].constituency_id.nunique(), 1)
        self.assertTrue(edges.source_voter_id.isin(voter_ids).all())
        self.assertTrue(edges.target_voter_id.isin(voter_ids).all())
        self.assertGreater(result["diagnostics"]["excluded_boundary_edge_count"], 0)

        events = result["events"]
        expected = pd.read_csv(DATA / "political_events.csv")
        region = result["diagnostics"]["region_id"]
        mask = (expected.constituency_id.eq(1) |
                (expected.constituency_id.isna() & expected.region_id.eq(region)) |
                (expected.constituency_id.isna() & expected.region_id.isna()))
        self.assertEqual(set(events.event_id), set(expected.loc[mask, "event_id"]))
        self.assertEqual(len(events), 26)  # 12 local + 6 regional + 8 state.

    def test_event_scopes_do_not_leak_between_constituencies(self):
        # Small, independently validated fixture: C1/C2 share R1; C3 is in R2.
        voters = pd.DataFrame({
            "voter_id": ["v1", "v2", "v3"], "constituency_id": [1, 2, 3],
            "region_id": [1, 1, 2], "initial_opinion": [0.0] * 3,
            "susceptibility": [0.5] * 3, "influence_power": [1.0] * 3,
            "confidence_threshold": [1.0] * 3, "turnout_propensity": [0.5] * 3,
            "event_sensitivity": [0.5] * 3,
        })
        candidates = pd.DataFrame({
            "candidate_id": ["c1", "c2", "c3"], "constituency_id": [1, 2, 3],
            "region_id": [1, 1, 2], "local_popularity": [0.5] * 3,
            **{f"position_{issue}": [0.] * 3 for issue in ("employment", "economy", "education", "healthcare", "agriculture")},
        })
        edges = pd.DataFrame(columns=["edge_id", "constituency_id", "region_id",
                                      "source_voter_id", "target_voter_id", "edge_weight"])
        events = pd.DataFrame({
            "event_id": ["local1", "local2", "local3", "region1", "region2", "state"],
            "region_id": pd.array([1, 1, 2, 1, 2, None], dtype="Int64"),
            "constituency_id": pd.array([1, 2, 3, None, None, None], dtype="Int64"),
            "event_day": [1] * 6, "duration": [1] * 6,
            "intensity": [0.5] * 6, "credibility": [0.5] * 6,
            "candidate_id": [None] * 6, "candidate_effect": [None] * 6,
            "source_type": ["News"] * 6, "issue": ["general"] * 6,
            "target_group": ["General"] * 6,
        })
        with TemporaryDirectory() as directory:
            for name, table in (("voters", voters), ("candidates", candidates),
                                ("social_network", edges), ("political_events", events)):
                table.to_csv(Path(directory) / f"{name}.csv", index=False)
            for constituency, expected_ids in (
                (1, {"local1", "region1", "state"}),
                (2, {"local2", "region1", "state"}),
                (3, {"local3", "region2", "state"}),
            ):
                with self.subTest(constituency=constituency):
                    result = load_constituency(directory, constituency)
                    self.assertEqual(set(result["events"].event_id), expected_ids)

    def test_unknown_constituency_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "was not found"):
            load_constituency(DATA, 999)


if __name__ == "__main__":
    unittest.main()
