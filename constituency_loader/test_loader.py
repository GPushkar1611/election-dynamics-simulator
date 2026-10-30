"""Small integration checks for the supplied constituency data."""

import unittest
from pathlib import Path

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

    def test_unknown_constituency_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "was not found"):
            load_constituency(DATA, 999)


if __name__ == "__main__":
    unittest.main()
