from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes.status import router as status_router
from api.routes.prediction import router as prediction_router
from api.routes.auth import router as auth_router

API_DESCRIPTION = """
Stock LSTM Prediction API exposes a trained recurrent neural network for next-day closing-price prediction.

Workflow:

1. Train the model with `python -m training.train --ticker AAPL`.
2. Start the API with `uvicorn api.main:app --reload`.
3. Send at least 60 historical closing prices to `POST /predict`.

Monitoring is available at `/metrics` for Prometheus-compatible collectors.
"""

TAGS_METADATA = [
    {
        "name": "Authentication",
        "description": "Token generation and verification endpoints.",
    },
    {
        "name": "Status",
        "description": "Readiness and health endpoints for deployment checks.",
    },
    {
        "name": "Prediction",
        "description": "Model inference endpoints for stock closing-price prediction.",
    },
    {
        "name": "Monitoring",
        "description": "Prometheus metrics exposed by prometheus-fastapi-instrumentator.",
    },
]

app = FastAPI(
    title="Stock LSTM Prediction API",
    summary="Predict the next stock closing price using a trained LSTM model.",
    description=API_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Tech Challenge Fase 4",
        "url": "https://github.com/",
    },
    license_info={
        "name": "Educational use",
    },
    openapi_tags=TAGS_METADATA,
)

# Include routers
app.include_router(auth_router)
app.include_router(status_router)
app.include_router(prediction_router)

# Expose metrics
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    tags=["Monitoring"],
)
