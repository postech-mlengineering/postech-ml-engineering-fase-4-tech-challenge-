from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset


class AppleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.fc(output[:, -1, :])


@dataclass(frozen=True)
class TrainingMetrics:
    mae: float
    rmse: float
    mape: float


def train_model(data: dict[str, torch.Tensor], epochs: int) -> tuple[AppleLSTM, TrainingMetrics]:
    model = AppleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(data["X_train"], data["y_train"]), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted = model(data["X_test"]).numpy().flatten()
        actual = data["y_test"].numpy().flatten()

    non_zero = actual != 0
    mape = float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100) if np.any(non_zero) else 0.0
    metrics = TrainingMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        mape=mape,
    )
    return model, metrics
