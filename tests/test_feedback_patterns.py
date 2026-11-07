import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import meok_neural_learning as learning

class FeedbackPatternsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        mock = patch.object(learning, "MEOK_DATA_DIR", Path(self.tmp.name))
        mock.start()
        self.addCleanup(mock.stop)
        self.predictor = learning.NeuralPredictor("test")

    def pattern(self, match, output="answer"):
        self.predictor._models["lookup"] = {"patterns": [{"match": match, "output": output}]}

    def data(self, rating=5, output="answer", count=50):
        return [{"input": {"a": 1}, "output": output, "rating": rating} for _ in range(count)]

    def test_empty_pattern_matches_only_empty_input(self):
        self.pattern({})
        self.assertIsNone(self.predictor.predict("lookup", {"unsafe": True}))
        self.assertEqual(self.predictor.predict("lookup", {}), "answer")

    def test_subset_cannot_ignore_changed_authority_context(self):
        self.pattern({"a": 1})
        self.assertIsNone(self.predictor.predict("lookup", {"a": 1, "scope": "different"}))

    def test_missing_key_is_not_null(self):
        self.pattern({"a": None})
        self.assertIsNone(self.predictor.predict("lookup", {}))

    def test_boolean_is_not_integer(self):
        self.pattern({"a": 1})
        self.assertIsNone(self.predictor.predict("lookup", {"a": True}))

    def test_key_order_does_not_change_identity(self):
        self.pattern({"a": 1, "b": {"c": 2, "d": 3}})
        self.assertEqual(self.predictor.predict("lookup", {"b": {"d": 3, "c": 2}, "a": 1}), "answer")

    def test_malformed_legacy_pattern_is_not_wildcard(self):
        self.predictor._models["lookup"] = {"patterns": [{"output": "unsafe"}]}
        self.assertIsNone(self.predictor.predict("lookup", {}))

    def test_non_dictionary_prediction_rejected(self):
        with self.assertRaises(ValueError): self.predictor.predict("lookup", [])

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError): self.predictor.predict("lookup", {"a": float("nan")})

    def test_nonstring_keys_cannot_alias_string_keys(self):
        self.pattern({"1": "x"})
        with self.assertRaises(ValueError): self.predictor.predict("lookup", {1: "x"})

    def test_tuple_cannot_alias_json_array(self):
        with self.assertRaises(ValueError): self.predictor.predict("lookup", {"nested": {"a": (1, 2)}})

    def test_logger_rejects_lossy_input_before_storage(self):
        logger = learning.InteractionLogger("test")
        for value in [{1: "x"}, {"a": object()}, {"nested": [{"x": (1, 2)}]}]:
            with self.assertRaises(ValueError): logger.log("lookup", value, {}, user_rating=5)
        self.assertEqual(logger.get_stats()["total_interactions"], 0)

    def test_training_rejects_lossy_input_before_activation(self):
        data = [dict(row, input={1: "x"}) for row in self.data()]
        with self.assertRaises(ValueError): self.predictor.train("lookup", data)
        self.assertFalse(self.predictor.ready)

    def test_unrated_examples_do_not_become_patterns(self):
        result = self.predictor.train("lookup", self.data(rating=None))
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["samples"], 0)
        self.assertFalse(self.predictor.ready)

    def test_higher_rating_wins_regardless_of_order(self):
        self.predictor.train("lookup", self.data(rating=3, output="old", count=49) + self.data(output="better", count=1))
        self.assertEqual(self.predictor.predict("lookup", {"a": 1}), "better")

    def test_equal_rating_conflict_requires_review(self):
        result = self.predictor.train("lookup", self.data(count=25) + self.data(output="conflict", count=25))
        self.assertEqual(result["status"], "no_usable_patterns")
        self.assertFalse(self.predictor.ready)

    def test_failed_build_preserves_existing_pattern(self):
        self.predictor.train("lookup", self.data())
        path = self.predictor.model_dir / "lookup.json"
        old = path.read_bytes()
        self.predictor.train("lookup", self.data(count=25) + self.data(output="conflict", count=25))
        self.assertEqual(path.read_bytes(), old)

    def test_corrections_are_not_silently_ignored(self):
        data = [dict(row, correction="This answer is wrong") for row in self.data()]
        self.assertEqual(self.predictor.train("lookup", data)["samples"], 0)

    def test_support_is_not_probability(self):
        self.predictor.train("lookup", self.data())
        model = json.loads((self.predictor.model_dir / "lookup.json").read_text())
        self.assertEqual(model["patterns"][0]["support_count"], 50)
        self.assertNotIn("confidence", model["patterns"][0])

    def test_logger_excludes_unrated_and_corrected(self):
        logger = learning.InteractionLogger("test")
        logger.log("lookup", {}, "unrated")
        logger.log("lookup", {}, "corrected", user_rating=5, user_correction="wrong")
        logger.log("lookup", {}, "accepted", user_rating=4)
        self.assertEqual(logger.get_training_data("lookup"), [{"input": {}, "output": "accepted", "rating": 4}])

    def test_invalid_ratings_rejected(self):
        logger = learning.InteractionLogger("test")
        for rating in [True, False, 0, 6, 99, 3.5, "5"]:
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                logger.log("lookup", {}, "answer", user_rating=rating)

    def test_ready_count_is_per_tool_and_eligible(self):
        logger = learning.InteractionLogger("test")
        for _ in range(49):
            logger.log("a", {}, "answer", user_rating=5)
            logger.log("b", {}, "answer", user_rating=5)
        for _ in range(10): logger.log("a", {}, "unrated")
        self.assertFalse(logger.get_stats()["ready_for_training"])
        logger.log("a", {}, "answer", user_rating=5)
        self.assertTrue(logger.get_stats()["ready_for_training"])

    def test_dashboard_rating_is_validated(self):
        _, _, rate = learning.get_learning_tools("test")
        with self.assertRaises(ValueError): rate("lookup", 100)

if __name__ == "__main__": unittest.main()
