from __future__ import annotations

from datetime import date, datetime, timezone

from services.cache_service import PredictionCache, prediction_cache
from services.model_runtime_service import ActiveModel, ModelRuntimeService


class ModelService:
    def __init__(self, runtime: ModelRuntimeService, cache: PredictionCache = prediction_cache) -> None:
        self._runtime = runtime
        self._cache = cache

    def train(self, ticker: str, start_date: date, lookback: int, epochs: int) -> ActiveModel:
        # Import data-science dependencies only for a training request. This keeps
        # health, authentication and status endpoints available on constrained hosts.
        from services.training.data_service import download_close_prices, normalize_ticker
        from services.training.feature_service import build_training_data
        from services.training.training_service import train_model

        self._runtime.start_training()
        try:
            normalized_ticker = normalize_ticker(ticker)
            close_prices = download_close_prices(normalized_ticker, start_date)
            data, scaler = build_training_data(close_prices, lookback)
            model, metrics = train_model(data, epochs)
            active_model = ActiveModel(
                model=model,
                scaler=scaler,
                ticker=normalized_ticker,
                trained_at=datetime.now(timezone.utc),
                lookback=lookback,
                metrics={"mae": metrics.mae, "rmse": metrics.rmse, "mape": metrics.mape},
            )
            self._runtime.activate(active_model)
            self._cache.clear()
            return active_model
        finally:
            self._runtime.finish_training()
