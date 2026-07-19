import os
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from api.application_services import ModelOperations, PredictionOperations
from services.cache_service import prediction_cache
from services.model_runtime_service import ModelRuntimeService
from services.model_service import ModelService
from services.prediction_use_case_service import PredictionService
from services.rate_limiter import predict_rate_limiter

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-lstm-prediction-challenge")
JWT_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

model_runtime = ModelRuntimeService()
model_service = ModelService(model_runtime, prediction_cache)
prediction_service = PredictionService(model_runtime)
model_operations = ModelOperations(model_service, model_runtime)
prediction_operations = PredictionOperations(prediction_service)

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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges are required.",
        )
    return current_user

def rate_limit_predict(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not predict_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Limit is 5 requests per minute.",
        )
