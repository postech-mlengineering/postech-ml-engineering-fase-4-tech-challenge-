from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock, RLock
from typing import Any


class ModelUnavailableError(RuntimeError):
    pass


class TrainingInProgressError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveModel:
    model: Any
    scaler: Any
    ticker: str
    trained_at: datetime
    lookback: int
    metrics: dict[str, float]


class ModelRuntimeService:
    """Process-local model state."""

    def __init__(self) -> None:
        self._state_lock = RLock()
        self._training_lock = Lock()
        self._active: ActiveModel | None = None

    def start_training(self) -> None:
        if not self._training_lock.acquire(blocking=False):
            raise TrainingInProgressError("A model training operation is already in progress.")

    def finish_training(self) -> None:
        self._training_lock.release()

    def activate(self, active_model: ActiveModel) -> None:
        with self._state_lock:
            self._active = active_model

    def get_active(self) -> ActiveModel:
        with self._state_lock:
            if self._active is None:
                raise ModelUnavailableError("No active model. An administrator must train a model first.")
            return self._active

    def status(self) -> ActiveModel | None:
        with self._state_lock:
            return self._active
