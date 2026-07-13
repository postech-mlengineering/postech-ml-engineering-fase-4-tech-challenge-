# Stock LSTM Prediction API (Backend)

This is the backend service for the Stock LSTM Prediction API. It exposes endpoints to check service health, readiness, Prometheus metrics, and run next-day closing price predictions using a trained LSTM model.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (A fast Python package installer and resolver)
- Python 3.11+

---

## Installation & Setup

We use `uv` to manage the virtual environment and dependencies.

### 1. Initialize & Create Virtual Environment
If you haven't already, initialize the virtual environment inside the `backend` folder:
```bash
uv venv
```

### 2. Activate the Virtual Environment
Activate the environment based on your operating system:

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\activate
  ```
- **Windows (cmd):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI application with Uvicorn:

```bash
uvicorn app:app --reload
```
Alternatively, you can run it directly using `uv run` (without manual activation):
```bash
uv run uvicorn app:app --reload
```

The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Interactive Swagger documentation will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Endpoint Details

### 1. Root Health Check (`GET /`)
Checks whether the server is up and if the required model artifacts are loaded.
- **Response Example:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 2. Deployment Health Check (`GET /health`)
Used for readiness checks in environments like Docker, Kubernetes, etc.
- **Response Example:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 3. Predict Next Closing Price (`POST /predict`)
Requires a history of at least 60 prices. Only the last 60 prices are used for inference.
- **Body Example:**
  ```json
  {
    "prices": [150.1, 151.2, ..., 160.4] 
  }
  ```
- **Response Example:**
  ```json
  {
    "prediction": 160.42
  }
  ```

### 4. Prometheus Metrics (`GET /metrics`)
Exposes performance metrics for Prometheus scraping.
