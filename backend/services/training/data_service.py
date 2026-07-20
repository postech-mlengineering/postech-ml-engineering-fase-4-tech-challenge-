
from __future__ import annotations

from datetime import date
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class TrainingDataError(ValueError):
    """Raised when market data cannot be used for training."""


class MarketDataProviderError(RuntimeError):
    """Raised when Yahoo Finance cannot be reached or queried successfully."""


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
            threads=False,
            timeout=15,
        )
    except Exception as exc:
        logger.exception("Yahoo Finance download failed for ticker=%s start_date=%s", normalized_ticker, start_date)
        raise MarketDataProviderError(
            "Could not reach Yahoo Finance. Retry shortly; if the problem persists, verify provider availability "
            "and outbound network access in the deployment environment."
        ) from exc

    if frame.empty or "Close" not in frame:
        logger.warning(
            "Yahoo Finance returned no usable data for ticker=%s start_date=%s rows=%s columns=%s",
            normalized_ticker,
            start_date,
            len(frame.index),
            list(frame.columns),
        )
        raise TrainingDataError(
            f"Yahoo Finance returned no closing-price data for {normalized_ticker} since {start_date}. "
            "Check the ticker and date range, then retry. If they are valid, this can indicate a temporary "
            "Yahoo Finance limitation or outbound-network restriction; check server logs for details."
        )

    close = frame["Close"]
    if getattr(close, "ndim", 1) == 2:
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        logger.warning("Yahoo Finance returned only missing closing prices for ticker=%s", normalized_ticker)
        raise TrainingDataError(
            f"Yahoo Finance returned only missing closing prices for {normalized_ticker}. "
            "Retry later or use a different ticker/date range."
        )
    return close
