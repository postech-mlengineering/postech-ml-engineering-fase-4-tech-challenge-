import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, "backend")

from services.cache_service import PredictionCache
from services.model_runtime_service import ActiveModel, ModelRuntimeService
from services.prediction_use_case_service import PredictionService


class PredictionServiceTests(unittest.TestCase):
    def test_uses_cache_after_first_prediction(self):
        runtime = ModelRuntimeService()
        runtime.activate(
            ActiveModel(
                model=object(),
                scaler=object(),
                ticker="AAPL",
                trained_at=datetime.now(timezone.utc),
                lookback=60,
                metrics={"mae": 0.1, "rmse": 0.2, "mape": 1.0},
            )
        )
        service = PredictionService(runtime, PredictionCache())
        prices = [100.0] * 60

        with patch("services.prediction_use_case_service.predict_next_close", return_value=123.456) as prediction:
            self.assertEqual(service.predict(prices), 123.456)
            self.assertEqual(service.predict(prices), 123.456)

        prediction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
