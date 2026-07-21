from datetime import date, datetime

from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(description="API process status.")
    model_loaded: bool = Field(description="Whether an active model is currently held in process memory.")

class PredictionRequest(BaseModel):
    prices: list[float] = Field(
        ...,
        min_length=60,
        description="At least 60 historical closing prices, ordered from oldest to newest.",
    )

class PredictionResponse(BaseModel):
    prediction: float


class UserAuthRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TrainModelRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15, description="Ticker to download and train, such as AAPL.")
    lookback: int = Field(default=60, ge=20, le=120, description="Number of closing prices in each LSTM input window.")
    epochs: int = Field(default=50, ge=1, le=50, description="Training epochs; capped to protect the Render Free instance.")


class ModelMetadata(BaseModel):
    ticker: str = Field(description="Ticker used for this in-memory model.")
    trained_at: datetime = Field(description="UTC time when training completed.")
    lookback: int = Field(description="Input window used while training and predicting.")
    metrics: dict[str, float] = Field(description="MAE, RMSE, and MAPE measured on normalized test data.")


class TrainModelResponse(BaseModel):
    status: str = "succeeded"
    model: ModelMetadata


class ActiveModelResponse(BaseModel):
    model_loaded: bool = Field(description="False when no model exists in the current API process.")
    model: ModelMetadata | None = Field(default=None, description="Metadata for the active in-memory model, if one exists.")
