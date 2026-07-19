from __future__ import annotations

from services.cache_service import PredictionCache, prediction_cache
from services.prediction_service import predict_next_close
from services.training.ml.artifact_service import LocalArtifactService


class PredictionService:
    """Coordinates local artifact loading, prediction, and caching."""

    def __init__(self, artifacts: LocalArtifactService, cache: PredictionCache = prediction_cache) -> None:
        self._artifacts = artifacts
        self._cache = cache

    def predict(self, prices: list[float]) -> float:
        cached_value = self._cache.get(prices)
        if cached_value is not None:
            return cached_value

        model, scaler, metadata = self._artifacts.load()
        prediction = predict_next_close(prices, model, scaler, metadata["lookback"])
        self._cache.set(prices, prediction)
        return prediction
