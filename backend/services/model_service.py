from __future__ import annotations

from datetime import date, datetime, timezone
import logging
from threading import Lock

from services.cache_service import PredictionCache, prediction_cache
from services.training.ml.artifact_service import LocalArtifactService

logger = logging.getLogger(__name__)


class TrainingInProgressError(RuntimeError):
    pass


class ModelService:
    def __init__(self, artifacts: LocalArtifactService, cache: PredictionCache = prediction_cache) -> None:
        self._artifacts = artifacts
        self._cache = cache
        self._training_lock = Lock()

    def train(self, ticker: str, start_date: date, lookback: int, epochs: int) -> dict:
        from services.training.data_service import download_close_prices, normalize_ticker
        from services.training.feature_service import build_training_data
        from services.training.training_service import train_model

        if not self._training_lock.acquire(blocking=False):
            raise TrainingInProgressError("A model training operation is already in progress.")
        try:
            normalized_ticker = normalize_ticker(ticker)
            logger.info("Model training started for ticker=%s", normalized_ticker)
            close_prices = download_close_prices(normalized_ticker, start_date)
            data, scaler = build_training_data(close_prices, lookback)
            model, metrics = train_model(data, epochs)
            metadata = {
                "ticker": normalized_ticker,
                "trained_at": datetime.now(timezone.utc),
                "lookback": lookback,
                "metrics": {"mae": metrics.mae, "rmse": metrics.rmse, "mape": metrics.mape},
            }
            self._artifacts.save(model, scaler, metadata)
            self._cache.clear()
            logger.info("Model training completed and artifacts were saved for ticker=%s", normalized_ticker)
            return metadata
        except ValueError as exc:
            logger.warning("Model training rejected for ticker=%s: %s", ticker, exc)
            raise
        except Exception:
            logger.exception("Model training failed for ticker=%s", ticker)
            raise
        finally:
            self._training_lock.release()
            logger.info("Model training lock released for ticker=%s", ticker)
