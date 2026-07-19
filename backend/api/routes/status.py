from fastapi import APIRouter, Depends
from api.dependencies import get_model_operations
from api.models.schemas import HealthResponse

router = APIRouter(tags=["Status"])

@router.get(
    "/",
    response_model=HealthResponse,
    summary="Root health check",
    description=(
        "Return API status and whether an active model is currently held in memory. "
        "`model_loaded` becomes false after a restart, sleep, or redeploy on Render Free."
    ),
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
def root(operations = Depends(get_model_operations)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=operations.is_loaded())


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Deployment health check",
    description=(
        "Return API readiness information and active in-memory model availability. Use this endpoint in Docker, "
        "Render, Railway, OCI, or load-balancer health checks."
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
def health(operations = Depends(get_model_operations)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=operations.is_loaded())
