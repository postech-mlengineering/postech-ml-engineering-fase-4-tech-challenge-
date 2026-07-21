# Stock LSTM Prediction API (Backend)

This is the backend service for the Stock LSTM Prediction API. It exposes endpoints to authenticate users, check service health, readiness, Prometheus metrics, and run next-day closing price predictions using a trained LSTM model.

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

The API will be available at: 

## Development
### API
[http://127.0.0.1:8000](http://127.0.0.1:8000)
Interactive Swagger documentation will be available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Production
### API
[https://api-132t.onrender.com](https://api-132t.onrender.com)
Interactive Swagger documentation will be available at: [https://api-132t.onrender.com/docs](https://api-132t.onrender.com/docs)

### Prometheus
Client to get metrics
[https://prometheus-132t.onrender.com/](https://prometheus-132t.onrender.com/)
Interactive Prometheus metrics will be available at: [https://prometheus-132t.onrender.com/metrics](https://prometheus-132t.onrender.com/metrics)

### Grafana
Dashboard
[https://grafana-132t.onrender.com/](https://grafana-132t.onrender.com/)
Interactive Grafana dashboard will be available at: [https://grafana-132t.onrender.com/](https://grafana-132t.onrender.com/)

---

## Postman Configuration

Import [postman_collection.json](postman_collection.json) file in Postman to get all endpoints ready.

---

## Endpoint Details

### 1. Authenticate User & Get Token (`POST /auth/token`)
Accepts JSON payload with credentials and returns a JWT access token valid for 30 minutes.
- **Request Body:**
  ```json
  {
    "username": "admin",
    "password": "admin"
  }
  ```
- **Response Example:**
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
    "token_type": "bearer"
  }
  ```

### 2. Root Health Check (`GET /`)
Checks whether the server is up and if the required model artifacts are loaded.
- **Response Example:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 3. Deployment Health Check (`GET /health`)
Used for readiness checks in environments like Docker, Kubernetes, etc.
- **Response Example:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 4. Predict Next Closing Price (`POST /predict`)
Requires a history of at least 60 prices. Only the last 60 prices are used for inference.
- **Security**: Requires a valid Bearer token in the `Authorization` header (`Authorization: Bearer <token>`).
- **Caching**: Response predictions are cached in-memory for 5 minutes using a hash of the input sequence.
- **Rate Limit**: Enforces a limit of 5 requests per minute per IP address. Exceeding this returns a `429 Too Many Requests` status code.
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

### 5. Prometheus Metrics (`GET /metrics`)
Exposes performance metrics for Prometheus scraping.

### 6. Train Model (`POST /ml/train`)
Requires an administrator Bearer token. Downloads closing prices for the requested ticker, trains the LSTM synchronously, and writes `model.pth`, `scaler.pkl`, and `metadata.json` to `backend/services/training/ml/artifacts`.
- **Security**: Requires a valid Bearer token for the administrator user.
- **Body Example:**
```json
{
  "ticker": "AAPL",
  "lookback": 60,
  "epochs": 50
}
```

On Render Free the local artifacts are lost whenever the service restarts, sleeps, or is redeployed; train again before calling `/predict`.
`API_ADMIN_USERNAME` can be set to a different administrator account; it defaults to `API_USERNAME`.

### 7. Active Model (`GET /ml/active`)
Returns whether the local model artifact set exists and, when available, its ticker, training timestamp, parameters, and metrics.
- **Security**: Requires a valid Bearer token for the administrator user.