"""Independent invalid-input regression cases for plan section 2.2."""

import unittest

import numpy as np
import pandas as pd

try:
    from .validation import DataValidationError, validate_data
except ImportError:
    from validation import DataValidationError, validate_data


def tables():
    voters = pd.DataFrame({
        "voter_id": ["v1", "v2"], "constituency_id": [1, 2], "region_id": [1, 2],
        "initial_opinion": [0., 0.], "susceptibility": [0.5, 0.5],
        "influence_power": [1., 1.], "confidence_threshold": [1., 1.],
        "turnout_propensity": [0.5, 0.5], "event_sensitivity": [0.5, 0.5],
        "occupation": ["Student", "Farmer"], "issue_economy": [0.5, 0.5],
    })
    candidates = pd.DataFrame({
        "candidate_id": ["c1", "c2"], "constituency_id": [1, 2], "region_id": [1, 2],
        "local_popularity": [0.5, 0.5],
        **{f"position_{issue}": [0., 0.] for issue in
           ("employment", "economy", "education", "healthcare", "agriculture")},
    })
    edges = pd.DataFrame({"edge_id": ["e1"], "constituency_id": [2], "region_id": [2],
                          "source_voter_id": ["v1"], "target_voter_id": ["v2"], "edge_weight": [1.]})
    events = pd.DataFrame({"event_id": ["news"], "constituency_id": [1], "region_id": [1],
                           "event_day": [1.], "duration": [1.], "intensity": [0.5],
                           "credibility": [0.5], "candidate_id": ["c1"], "candidate_effect": [0.1],
                           "source_type": ["News"], "issue": ["general"], "target_group": ["General"]})
    return voters, edges, candidates, events


class ValidationTests(unittest.TestCase):
    def test_valid_fixture(self):
        validate_data(*tables())

    def test_invalid_numeric_values_and_categories(self):
        for index, column, value in (
            (0, "influence_power", np.inf), (1, "edge_weight", 1.1),
            (1, "edge_weight", np.inf), (0, "constituency_id", 1.5),
            (0, "occupation", "Unknown"), (0, "issue_economy", -0.1),
            (2, "position_economy", 1.1), (3, "event_day", 31.),
            (3, "event_day", 1.5), (3, "duration", 1.5), (3, "duration", np.inf),
            (3, "source_type", "Unknown"), (3, "issue", "Unknown"),
            (3, "target_group", "Unknown"), (3, "event_id", ""),
        ):
            with self.subTest(column=column, value=value):
                data = tables()
                # Use object for malformed identifiers without pandas dtype coercion errors.
                data[index][column] = data[index][column].astype(object)
                data[index].loc[0, column] = value
                with self.assertRaises(DataValidationError):
                    validate_data(*data)

    def test_unknown_region_and_candidate_scope(self):
        data = tables()
        data[3].loc[0, "constituency_id"] = np.nan
        data[3].loc[0, "region_id"] = 99
        with self.assertRaisesRegex(DataValidationError, "region_id does not exist"):
            validate_data(*data)
        data = tables()
        data[3].loc[0, "candidate_id"] = "c2"
        with self.assertRaisesRegex(DataValidationError, "event candidate must belong"):
            validate_data(*data)

    def test_candidate_effect_and_source_consistency(self):
        for changes in ({"candidate_id": None}, {"candidate_effect": None},
                        {"source_type": "External"},
                        {"source_type": "Candidate", "candidate_id": None, "candidate_effect": None},
                        {"region_id": None, "constituency_id": None}):
            with self.subTest(changes=changes):
                data = tables()
                for column, value in changes.items():
                    data[3][column] = data[3][column].astype(object)
                    data[3].loc[0, column] = value
                with self.assertRaises(DataValidationError):
                    validate_data(*data)

    def test_duplicate_ids_and_missing_schema_report_validation_errors(self):
        for index, column in ((0, "voter_id"), (2, "candidate_id")):
            data = tables()
            data[index].loc[1, column] = data[index].loc[0, column]
            with self.assertRaises(DataValidationError):
                validate_data(*data)
        data = tables()
        data[2].drop(columns="position_economy", inplace=True)
        with self.assertRaisesRegex(DataValidationError, "missing columns"):
            validate_data(*data)

    def test_duration_may_extend_past_close_and_custom_calendar_is_explicit(self):
        data = tables()
        data[3].loc[0, "event_day"] = 30
        data[3].loc[0, "duration"] = 5
        validate_data(*data)  # The scheduler will truncate effects at day 30.
        data[3].loc[0, "event_day"] = 60
        validate_data(*data, election_duration_days=60)
        with self.assertRaises(DataValidationError):
            validate_data(*data)

    def test_optional_candidate_and_event_metadata_is_checked(self):
        for index, column, value in ((2, "experience_level", 6), (2, "is_incumbent", "yes"),
                                      (3, "event_type", "Unknown")):
            data = tables()
            data[index][column] = value
            with self.subTest(column=column), self.assertRaises(DataValidationError):
                validate_data(*data)
        data = tables()
        data = (data[0], data[1], data[2].iloc[:1], data[3])
        with self.assertRaisesRegex(DataValidationError, "at least one candidate"):
            validate_data(*data)


if __name__ == "__main__":
    unittest.main()
