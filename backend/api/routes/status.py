from fastapi import APIRouter
from api.schemas import HealthResponse
from services.model_loader import artifacts_available

router = APIRouter(tags=["Status"])

@router.get(
    "/",
    response_model=HealthResponse,
    summary="Root health check",
    description="Return a simple API status and whether the trained model artifacts are present.",
    responses={
        200: {
            "description": "API status returned successfully.",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "model_loaded": True}
                }
            },
        }
    },
)
def root() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=artifacts_available())


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Deployment health check",
    description=(
        "Return API readiness information. Use this endpoint in Docker, Railway, OCI or load balancer "
        "health checks."
    ),
    responses={
        200: {
            "description": "API status returned successfully.",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "model_loaded": False}
                }
            },
        }
    },
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=artifacts_available())
