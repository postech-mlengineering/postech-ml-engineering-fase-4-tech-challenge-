from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_model_artifacts, get_current_user, rate_limit_predict
from api.schemas import PredictionRequest, PredictionResponse
from services.prediction_service import predict_next_close
from services.cache_service import prediction_cache

router = APIRouter(tags=["Prediction"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict next closing price",
    description=(
        "Predict the next closing price from historical closes. The request must contain at least 60 prices "
        "ordered from oldest to newest. If more than 60 prices are provided, only the latest 60 are used. "
        "This endpoint is rate-limited and requires JWT authentication."
    ),
    response_description="Predicted next closing price rounded to two decimal places.",
    dependencies=[Depends(rate_limit_predict)],
    responses={
        200: {
            "description": "Prediction generated successfully.",
            "content": {
                "application/json": {
                    "example": {"prediction": 160.42}
                }
            },
        },
        401: {
            "description": "Unauthorized, JWT token is missing or invalid.",
        },
        422: {
            "description": "Validation error, usually because fewer than 60 prices were provided.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "too_short",
                                "loc": ["body", "prices"],
                                "msg": "List should have at least 60 items after validation, not 3",
                            }
                        ]
                    }
                }
            },
        },
        429: {
            "description": "Too many requests. Rate limit exceeded.",
        },
        503: {
            "description": "Model artifacts are missing. Train the model before calling this endpoint.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "Missing artifacts. Expected model at model/stock_lstm.keras "
                            "and scaler at model/scaler.pkl."
                        )
                    }
                }
            },
        },
    },
)
def predict(
    request: PredictionRequest,
    artifacts=Depends(get_model_artifacts),
    current_user: str = Depends(get_current_user),
) -> PredictionResponse:
    # Check cache first
    cached_val = prediction_cache.get(request.prices)
    if cached_val is not None:
        return PredictionResponse(prediction=round(cached_val, 2))

    model, scaler = artifacts
    try:
        prediction = predict_next_close(request.prices, model, scaler)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    # Store in cache
    prediction_cache.set(request.prices, prediction)

    return PredictionResponse(prediction=round(prediction, 2))
