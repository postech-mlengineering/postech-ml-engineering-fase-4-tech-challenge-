import os
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from services.model_loader import load_model_and_scaler
from services.rate_limiter import predict_rate_limiter

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-lstm-prediction-challenge")
JWT_ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def get_model_artifacts() -> tuple[any, any]:
    try:
        return load_model_and_scaler()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

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

def rate_limit_predict(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not predict_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Limit is 5 requests per minute.",
        )
