from datetime import datetime, timedelta, timezone
import os
import jwt
from fastapi import APIRouter, HTTPException, status
from api.models.schemas import TokenResponse, UserAuthRequest

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-key-lstm-prediction-challenge")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin")

router = APIRouter(tags=["Authentication"])

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

@router.post(
    "/auth/token",
    response_model=TokenResponse,
    summary="Authenticate and get access token",
    description="Pass username and password in the JSON body to receive a JWT access token.",
)
def login(payload: UserAuthRequest) -> TokenResponse:
    if payload.username != API_USERNAME or payload.password != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": payload.username}, expires_delta=access_token_expires
    )
    return TokenResponse(access_token=access_token, token_type="bearer")
