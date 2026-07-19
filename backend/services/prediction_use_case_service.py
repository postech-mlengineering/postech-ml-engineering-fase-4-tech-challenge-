from __future__ import annotations

from services.cache_service import PredictionCache, prediction_cache
from services.model_runtime_service import ModelRuntimeService
from services.prediction_service import predict_next_close


class PredictionService:
    """Application service that coordinates active-model inference and caching."""

    def __init__(
        self,
        runtime: ModelRuntimeService,
        cache: PredictionCache = prediction_cache,
    ) -> None:
        self._runtime = runtime
        self._cache = cache

    def predict(self, prices: list[float]) -> float:
        cached_value = self._cache.get(prices)
        if cached_value is not None:
            return cached_value

        active_model = self._runtime.get_active()
        prediction = predict_next_close(
            prices,
            active_model.model,
            active_model.scaler,
            active_model.lookback,
        )
        self._cache.set(prices, prediction)
        return prediction
