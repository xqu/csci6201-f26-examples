"""Week 9: simple assertions plus two unittest cases."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from sklearn.datasets import load_iris

module_path = Path(__file__).with_name("02_explore_iris.py")
spec = importlib.util.spec_from_file_location("explore_iris", module_path)
explore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explore)


class TestIrisHelpers(unittest.TestCase):
    def test_summary(self):
        stats = explore.summarize_column([1, 2, 3])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["mean"], 2.0)

    def test_count_above(self):
        self.assertEqual(explore.count_values_above([1, 5, 5, 7], 5), 1)

    def test_dataset_size(self):
        assert load_iris().data.shape == (150, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
