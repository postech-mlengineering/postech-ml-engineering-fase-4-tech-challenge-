from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineeringError(ValueError):
    """Raised when there is not enough data to create training windows."""


def build_training_data(close_prices, lookback: int) -> tuple[dict[str, torch.Tensor], MinMaxScaler]:
    values = np.asarray(close_prices, dtype=np.float64).reshape(-1, 1)
    if len(values) <= lookback + 1:
        raise FeatureEngineeringError(
            f"At least {lookback + 2} closing prices are required to train this model."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    windows = np.asarray([scaled[index : index + lookback] for index in range(len(scaled) - lookback)])
    targets = np.asarray([scaled[index + lookback] for index in range(len(scaled) - lookback)])

    split = int(len(windows) * 0.8)
    if split == 0 or split == len(windows):
        raise FeatureEngineeringError("There is not enough data to create train and test sets.")

    return {
        "X_train": torch.tensor(windows[:split], dtype=torch.float32),
        "y_train": torch.tensor(targets[:split], dtype=torch.float32),
        "X_test": torch.tensor(windows[split:], dtype=torch.float32),
        "y_test": torch.tensor(targets[split:], dtype=torch.float32),
    }, scaler
