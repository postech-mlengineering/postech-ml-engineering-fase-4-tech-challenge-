from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import torch

class ArtifactUnavailableError(RuntimeError):
    """Raised when a complete local model artifact set is not available."""


class LocalArtifactService:
    """Stores the single active model version on the local service filesystem."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self._artifact_dir = artifact_dir or Path(__file__).resolve().parent / "artifacts"
        self._model_path = self._artifact_dir / "model.pth"
        self._scaler_path = self._artifact_dir / "scaler.pkl"
        self._metadata_path = self._artifact_dir / "metadata.json"
        self._lock = RLock()

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    def save(self, model: Any, scaler: Any, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            model_tmp = self._model_path.with_suffix(".pth.tmp")
            scaler_tmp = self._scaler_path.with_suffix(".pkl.tmp")
            metadata_tmp = self._metadata_path.with_suffix(".json.tmp")
            torch.save(model.state_dict(), model_tmp)
            joblib.dump(scaler, scaler_tmp)
            metadata_tmp.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
            model_tmp.replace(self._model_path)
            scaler_tmp.replace(self._scaler_path)
            metadata_tmp.replace(self._metadata_path)

    def load(self) -> tuple[Any, Any, dict[str, Any]]:
        with self._lock:
            if not self.available():
                raise ArtifactUnavailableError(
                    "No local model artifacts are available. An administrator must call POST /ml/train first."
                )
            from services.training.training_service import AppleLSTM

            model = AppleLSTM()
            state_dict = torch.load(self._model_path, map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model, joblib.load(self._scaler_path), json.loads(self._metadata_path.read_text(encoding="utf-8"))

    def metadata(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.available():
                return None
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))

    def available(self) -> bool:
        return self._model_path.is_file() and self._scaler_path.is_file() and self._metadata_path.is_file()


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
