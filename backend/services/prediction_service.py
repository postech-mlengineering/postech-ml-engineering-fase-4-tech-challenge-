import numpy as np
import torch
import torch.nn as nn

def predict_next_close(prices: list[float], model: any, scaler: any, lookback: int = 60) -> float:
    # Use the latest 60 prices
    prices_input = prices[-lookback:]
    if len(prices_input) < lookback:
        raise ValueError(f"List should have at least {lookback} items after validation.")

    # Format prices as a 2D numpy array of shape (60, 1)
    data = np.array(prices_input).reshape(-1, 1)
    data_scaled = scaler.transform(data)

    if isinstance(model, nn.Module):
        # PyTorch inference
        # Convert to tensor of shape (1, 60, 1)
        X_input = torch.FloatTensor(data_scaled).view(1, lookback, 1)
        with torch.no_grad():
            y_pred_norm = model(X_input)
        predicted_norm = y_pred_norm.numpy()
    else:
        # Fallback to Keras / generic model inference
        # Convert to shape (1, 60, 1)
        X_input = data_scaled.reshape(1, lookback, 1)
        y_pred_norm = model.predict(X_input)
        predicted_norm = y_pred_norm

    # Inverse scale prediction
    price = scaler.inverse_transform(predicted_norm)
    return float(price[0][0])
