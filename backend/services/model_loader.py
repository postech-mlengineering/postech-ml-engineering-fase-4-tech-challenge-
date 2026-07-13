import os
import joblib
import torch
import torch.nn as nn

class AppleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Define search paths for artifacts
POSSIBLE_PATHS = [
    # root level model/
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model")),
    # backend/model/
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")),
    # fallback to notebooks/models/
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "models")),
]

def get_artifact_paths() -> tuple[str | None, str | None]:
    model_path = None
    scaler_path = None

    for base_dir in POSSIBLE_PATHS:
        # Check for PyTorch model
        m_path_pth = os.path.join(base_dir, "model.pth")
        # Check for Keras model
        m_path_keras = os.path.join(base_dir, "stock_lstm.keras")
        s_path = os.path.join(base_dir, "scaler.pkl")

        if os.path.exists(m_path_pth) and os.path.exists(s_path):
            model_path = m_path_pth
            scaler_path = s_path
            break
        elif os.path.exists(m_path_keras) and os.path.exists(s_path):
            model_path = m_path_keras
            scaler_path = s_path
            break

    return model_path, scaler_path

def artifacts_available() -> bool:
    model_path, scaler_path = get_paths_to_verify()
    return model_path is not None and scaler_path is not None

def get_paths_to_verify() -> tuple[str | None, str | None]:
    # We also check if model/ directory is explicitly expected by app.py error message
    # Expected model at model/stock_lstm.keras and scaler at model/scaler.pkl
    # Let's check first if get_artifact_paths finds anything.
    model_path, scaler_path = get_artifact_paths()
    if model_path and scaler_path:
        return model_path, scaler_path
    
    # Fallback to local 'model' dir
    local_model_dir = os.path.abspath("model")
    m_path_keras = os.path.join(local_model_dir, "stock_lstm.keras")
    s_path = os.path.join(local_model_dir, "scaler.pkl")
    if os.path.exists(m_path_keras) and os.path.exists(s_path):
        return m_path_keras, s_path
        
    return None, None

def load_model_and_scaler() -> tuple[any, any]:
    model_path, scaler_path = get_paths_to_verify()
    if not model_path or not scaler_path:
        raise FileNotFoundError(
            "Missing artifacts. Expected model at model/stock_lstm.keras "
            "and scaler at model/scaler.pkl."
        )

    scaler = joblib.load(scaler_path)

    if model_path.endswith(".keras"):
        # If they load a Keras model, they would need tensorflow. Let's do lazy import.
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        # Load PyTorch model
        model = AppleLSTM()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

    return model, scaler
