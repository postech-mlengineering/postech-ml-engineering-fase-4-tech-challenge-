import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "backend")

from services.training.ml.artifact_service import LocalArtifactService


class FakeModel:
    def state_dict(self):
        return {"weights": [1, 2, 3]}


class LocalArtifactServiceTests(unittest.TestCase):
    def test_saves_complete_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = LocalArtifactService(Path(directory))
            artifacts.save(
                FakeModel(),
                {"scaler": "test"},
                {
                    "ticker": "AAPL",
                    "trained_at": datetime.now(timezone.utc),
                    "lookback": 60,
                    "metrics": {"mae": 0.1, "rmse": 0.2, "mape": 1.0},
                },
            )
            self.assertTrue(artifacts.available())
            self.assertTrue((Path(directory) / "model.pth").is_file())
            self.assertTrue((Path(directory) / "scaler.pkl").is_file())
            self.assertEqual(artifacts.metadata()["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
