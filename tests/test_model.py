import unittest
import pandas as pd
from src.model import PerformancePredictor


class TestModel(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            "Hours Studied": [1, 2, 3],
            "Previous Scores": [50, 60, 70],
            "Sleep Hours": [6, 7, 8],
            "Sample Question Papers Practiced": [1, 2, 3],
            "Extracurricular Activities": [1, 0, 1]
        })

        self.y = pd.Series([55, 65, 75])

        self.model = PerformancePredictor()

    def test_training(self):
        trained_model = self.model.train(self.X, self.y)
        self.assertIsNotNone(trained_model)

    def test_prediction(self):
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X)

        self.assertEqual(len(preds), len(self.X))

        # Ensure predictions are numeric
        self.assertTrue(all(isinstance(p, (int, float)) for p in preds))


if __name__ == "__main__":
    unittest.main()