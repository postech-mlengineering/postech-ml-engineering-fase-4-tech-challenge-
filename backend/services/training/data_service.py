
from __future__ import annotations

from datetime import date
import re
from typing import Any

class TrainingDataError(ValueError):
    """Raised when market data cannot be used for training."""


TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=-]{1,15}$")


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(normalized):
        raise TrainingDataError("Ticker must contain 1 to 15 valid market-symbol characters.")
    return normalized


def validate_start_date(start_date: date) -> None:
    today = date.today()
    if start_date >= today:
        raise TrainingDataError("Start date must be before today.")
    if (today - start_date).days > 3653:
        raise TrainingDataError("Start date must be within the last 10 years on the Render Free plan.")


def download_close_prices(ticker: str, start_date: date) -> Any:
    normalized_ticker = normalize_ticker(ticker)
    validate_start_date(start_date)
    try:
        import yfinance as yf

        frame = yf.download(
            normalized_ticker,
            start=start_date.isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        raise TrainingDataError(f"Could not download historical market data.") from exc

    if frame.empty or "Close" not in frame:
        raise TrainingDataError("No closing-price data was found for this ticker and date range.")

    close = frame["Close"]
    if getattr(close, "ndim", 1) == 2:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise TrainingDataError("The downloaded market data has no valid closing prices.")
    return close
