import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "backend")

from services.cache_service import PredictionCache
from services.prediction_use_case_service import PredictionService


class FakeArtifacts:
    def load(self):
        return object(), object(), {"lookback": 60}


class PredictionServiceTests(unittest.TestCase):
    def test_uses_cache_after_first_prediction(self):
        service = PredictionService(FakeArtifacts(), PredictionCache())
        prices = [100.0] * 60

        with patch("services.prediction_use_case_service.predict_next_close", return_value=123.456) as prediction:
            self.assertEqual(service.predict(prices), 123.456)
            self.assertEqual(service.predict(prices), 123.456)

        prediction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
