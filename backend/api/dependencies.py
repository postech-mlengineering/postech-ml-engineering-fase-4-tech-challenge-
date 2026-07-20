import os

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from api.models.schemas import ActiveModelResponse, ModelMetadata
from services.cache_service import prediction_cache
from services.model_service import ModelService, TrainingInProgressError
from services.prediction_use_case_service import PredictionService
from services.rate_limiter import predict_rate_limiter
from services.training.data_service import MarketDataProviderError
from services.training.ml.artifact_service import ArtifactUnavailableError, LocalArtifactService

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-lstm-prediction-challenge")
JWT_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

artifact_service = LocalArtifactService()
model_service = ModelService(artifact_service, prediction_cache)
prediction_service = PredictionService(artifact_service, prediction_cache)


class PredictionOperations:
    def predict(self, prices: list[float]) -> float:
        try:
            return prediction_service.predict(prices)
        except ArtifactUnavailableError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


class ModelOperations:
    def train(self, ticker: str, start_date, lookback: int, epochs: int) -> ModelMetadata:
        try:
            metadata = model_service.train(ticker, start_date, lookback, epochs)
        except TrainingInProgressError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except MarketDataProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model training failed.") from exc
        return ModelMetadata(**metadata)

    def active(self) -> ActiveModelResponse:
        metadata = artifact_service.metadata()
        if metadata is None:
            return ActiveModelResponse(model_loaded=False)
        return ActiveModelResponse(model_loaded=True, model=ModelMetadata(**metadata))

    def is_loaded(self) -> bool:
        return artifact_service.available()


model_operations = ModelOperations()
prediction_operations = PredictionOperations()


def get_model_operations() -> ModelOperations:
    return model_operations


def get_prediction_operations() -> PredictionOperations:
    return prediction_operations


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    return username


def get_current_admin(current_user: str = Depends(get_current_user)) -> str:
    admin_username = os.getenv("API_ADMIN_USERNAME", os.getenv("API_USERNAME", "admin"))
    if current_user != admin_username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator privileges are required.")
    return current_user


def rate_limit_predict(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not predict_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Limit is 5 requests per minute.",
        )
