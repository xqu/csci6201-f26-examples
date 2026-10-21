"""Week 9: demonstrate tests for reusable data processing helpers."""
import importlib.util
from pathlib import Path
import unittest
from sklearn.datasets import load_diabetes

module_path = Path(__file__).with_name("02_explore_diabetes.py")
spec = importlib.util.spec_from_file_location("explore_diabetes", module_path)
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


class TestDiabetesHelpers(unittest.TestCase):
    def test_summary(self):
        stats = explore.summarize_column([2, 4, 6])
        self.assertEqual(stats["mean"], 4.0)
        self.assertEqual(stats["min"], 2.0)

    def test_count_above(self):
        self.assertEqual(explore.count_above([100, 200, 201], 200), 1)

    def test_dataset_shape(self):
        assert load_diabetes().data.shape == (442, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
