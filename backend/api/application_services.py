"""HTTP-facing adapters around the domain services.

Routes depend on these operations through ``api.dependencies`` and do not
import concrete implementations from ``services``.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from .models.schemas import ActiveModelResponse, ModelMetadata
from services.model_runtime_service import ModelRuntimeService, ModelUnavailableError, TrainingInProgressError
from services.model_service import ModelService
from services.prediction_use_case_service import PredictionService


def _metadata(active) -> ModelMetadata:
    return ModelMetadata(
        ticker=active.ticker,
        trained_at=active.trained_at,
        lookback=active.lookback,
        metrics=active.metrics,
    )


class PredictionOperations:
    def __init__(self, prediction_service: PredictionService) -> None:
        self._prediction_service = prediction_service

    def predict(self, prices: list[float]) -> float:
        try:
            return self._prediction_service.predict(prices)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


class ModelOperations:
    def __init__(self, model_service: ModelService, runtime: ModelRuntimeService) -> None:
        self._model_service = model_service
        self._runtime = runtime

    def train(self, ticker, start_date, lookback: int, epochs: int) -> ModelMetadata:
        try:
            active = self._model_service.train(ticker, start_date, lookback, epochs)
        except TrainingInProgressError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model training failed. {exc}") from exc
        return _metadata(active)

    def active(self) -> ActiveModelResponse:
        active = self._runtime.status()
        if active is None:
            return ActiveModelResponse(model_loaded=False)
        return ActiveModelResponse(model_loaded=True, model=_metadata(active))

    def is_loaded(self) -> bool:
        return self._runtime.status() is not None
