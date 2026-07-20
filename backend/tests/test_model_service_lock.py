import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "backend")

from services.cache_service import PredictionCache
from services.model_service import ModelService
from services.training.data_service import TrainingDataError
from services.training.ml.artifact_service import LocalArtifactService


class ModelServiceLockTests(unittest.TestCase):
    def test_download_failure_releases_training_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ModelService(LocalArtifactService(Path(directory)), PredictionCache())
            with patch(
                "services.training.data_service.download_close_prices",
                side_effect=TrainingDataError("No data returned"),
            ):
                with self.assertRaises(TrainingDataError):
                    service.train("AAPL", date(2025, 1, 1), 60, 1)

            self.assertTrue(service._training_lock.acquire(blocking=False))
            service._training_lock.release()


if __name__ == "__main__":
    unittest.main()
