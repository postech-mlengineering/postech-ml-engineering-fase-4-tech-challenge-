from fastapi import APIRouter, Depends, status

from api.dependencies import get_current_admin, get_model_operations
from api.models.schemas import ActiveModelResponse, TrainModelRequest, TrainModelResponse


router = APIRouter(prefix="/ml", tags=["ML"])


@router.post(
    "/train",
    response_model=TrainModelResponse,
    status_code=status.HTTP_200_OK,
    summary="Train and save local LSTM artifacts",
    description=(
        "Downloads historical closing prices for the requested ticker, trains a PyTorch LSTM synchronously, "
        "and saves `model.pth`, `scaler.pkl`, and `metadata.json` under `backend/services/training/ml/artifacts`. "
        "Requires an administrator JWT. The local filesystem is ephemeral on Render Free, so these files are lost "
        "after a restart, sleep, or redeploy."
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
    summary="Get local model artifact status",
    description=(
        "Returns metadata for the local artifact set. `model_loaded` is false when the local files do not exist, "
        "including after a restart, sleep, or redeploy on Render Free."
    ),
)
def active_model(operations = Depends(get_model_operations)) -> ActiveModelResponse:
    return operations.active()
