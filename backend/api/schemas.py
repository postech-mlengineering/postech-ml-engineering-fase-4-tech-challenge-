from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictionRequest(BaseModel):
    prices: list[float] = Field(..., min_length=60, description="List of at least 60 historical closing prices ordered from oldest to newest.")

class PredictionResponse(BaseModel):
    prediction: float
