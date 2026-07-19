from fastapi import APIRouter, Depends, status

from api.dependencies import get_current_admin, get_model_operations
from api.models.schemas import ActiveModelResponse, TrainModelRequest, TrainModelResponse


router = APIRouter(prefix="/ml", tags=["ML"])


@router.post(
    "/train",
    response_model=TrainModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Train and activate an in-memory LSTM model",
    description=(
        "Downloads historical closing prices for the requested ticker, trains a PyTorch LSTM synchronously, "
        "and makes it the active model for this API process. Requires an administrator JWT. "
        "No `.pkl` or `.pth` artifact is persisted: the model and scaler are lost after a restart, sleep, or redeploy."
    ),
    response_description="The newly trained active model, including normalized-data evaluation metrics.",
    responses={
        401: {"description": "Missing or invalid JWT."},
        403: {"description": "The authenticated user is not the configured administrator."},
        409: {"description": "Another model-training operation is already running."},
        422: {"description": "Invalid ticker, unsupported date range, insufficient data, or invalid parameters."},
        500: {"description": "The model-training operation failed unexpectedly."},
    },
)
def train(
    request: TrainModelRequest,
    _: str = Depends(get_current_admin),
    operations = Depends(get_model_operations),
) -> TrainModelResponse:
    return TrainModelResponse(
        model=operations.train(request.ticker, request.start_date, request.lookback, request.epochs)
    )


@router.get(
    "/active",
    response_model=ActiveModelResponse,
    summary="Get active in-memory model status",
    description=(
        "Returns metadata for the model currently held in this API process. `model_loaded` is false after startup, "
        "sleep, restart, or redeploy until an administrator trains a new model."
    ),
)
def active_model(operations = Depends(get_model_operations)) -> ActiveModelResponse:
    return operations.active()
