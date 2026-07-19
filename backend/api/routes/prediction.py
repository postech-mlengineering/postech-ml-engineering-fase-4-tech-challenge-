from fastapi import APIRouter, Depends
from api.dependencies import get_current_user, get_prediction_operations, rate_limit_predict
from api.models.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["Prediction"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict next closing price",
    description=(
        "Predict the next closing price from historical closes. The request must contain at least 60 prices "
        "ordered from oldest to newest. If more than 60 prices are provided, only the latest 60 are used. "
        "This endpoint is rate-limited and requires JWT authentication. It uses only the active in-memory model; "
        "call `POST /ml/train` again after the service restarts, sleeps, or is redeployed."
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
            "description": "No active in-memory model is available. An administrator must call `POST /ml/train` first.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No active model. An administrator must train a model first."
                    }
                }
            },
        },
    },
)
def predict(
    request: PredictionRequest,
    current_user: str = Depends(get_current_user),
    operations = Depends(get_prediction_operations),
) -> PredictionResponse:
    prediction = operations.predict(request.prices)

    return PredictionResponse(prediction=round(prediction, 2))
