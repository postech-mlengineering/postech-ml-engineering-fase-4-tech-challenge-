from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes.status import router as status_router
from api.routes.prediction import router as prediction_router
from api.routes.auth import router as auth_router
from api.routes.ml import router as models_router

API_DESCRIPTION = """
Stock LSTM Prediction API exposes a trained recurrent neural network for next-day closing-price prediction.

Workflow:

1. Authenticate as an administrator and call `POST /ml/train` with a ticker.
2. The API downloads market data, trains a PyTorch LSTM, and writes the model, scaler, and metadata locally.
3. Send at least 60 historical closing prices to `POST /predict`.

## Render Free behavior

Training creates `model.pth`, `scaler.pkl`, and `metadata.json` in `backend/services/training/ml/artifacts`.
On Render Free this local filesystem is ephemeral: the files are lost when the service restarts, sleeps, or is
redeployed. In that case, call `POST /ml/train` again before requesting a prediction.

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
        "name": "ML",
        "description": (
            "Administrator-only training and local artifact status. Artifacts are lost after a restart, sleep, "
            "or redeploy on Render Free."
        ),
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
app.include_router(models_router)

# Expose metrics
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    tags=["Monitoring"],
)
