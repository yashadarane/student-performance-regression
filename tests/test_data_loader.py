import unittest
import pandas as pd
from src.data_loader import load_data


class TestDataLoader(unittest.TestCase):

    def test_load_valid_file(self):
        df = load_data("Student_Performance.csv")
        self.assertIsInstance(df, pd.DataFrame)

    def test_invalid_path(self):
        with self.assertRaises(RuntimeError):
            load_data("invalid/path.csv")


if __name__ == "__main__":
    unittest.main()