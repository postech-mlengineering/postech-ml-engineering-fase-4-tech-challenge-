import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, "backend")

from services.model_runtime_service import ActiveModel, ModelRuntimeService, ModelUnavailableError, TrainingInProgressError
from services.training.data_service import TrainingDataError, normalize_ticker, validate_start_date


class ModelRuntimeServiceTests(unittest.TestCase):
    def test_runtime_exposes_active_model(self):
        runtime = ModelRuntimeService()
        with self.assertRaises(ModelUnavailableError):
            runtime.get_active()

        active = ActiveModel(
            model=object(),
            scaler=object(),
            ticker="AAPL",
            trained_at=datetime.now(timezone.utc),
            lookback=60,
            metrics={"mae": 0.1, "rmse": 0.2, "mape": 1.0},
        )
        runtime.activate(active)
        self.assertEqual(runtime.get_active().ticker, "AAPL")

    def test_runtime_rejects_parallel_training(self):
        runtime = ModelRuntimeService()
        runtime.start_training()
        try:
            with self.assertRaises(TrainingInProgressError):
                runtime.start_training()
        finally:
            runtime.finish_training()

    def test_training_input_validation(self):
        self.assertEqual(normalize_ticker(" aapl "), "AAPL")
        with self.assertRaises(TrainingDataError):
            normalize_ticker("AAPL; DROP")
        with self.assertRaises(TrainingDataError):
            validate_start_date(datetime.now(timezone.utc).date())


if __name__ == "__main__":
    unittest.main()
