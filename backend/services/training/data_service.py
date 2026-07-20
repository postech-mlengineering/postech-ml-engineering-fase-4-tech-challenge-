
from __future__ import annotations

from datetime import date
import logging
import math
import os
import re
import time
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class TrainingDataError(ValueError):
    """Raised when market data cannot be used for training."""


class MarketDataProviderError(RuntimeError):
    """Raised when Yahoo Finance cannot be reached or queried successfully."""


class YahooRateLimitError(MarketDataProviderError):
    """Raised when Yahoo Finance asks this service to slow down."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            "Yahoo Finance is rate-limiting market-data requests. "
            f"Retry after approximately {retry_after_seconds} seconds."
        )


class MarketDataCache:
    """Process-local cache and provider cooldown for the Render Free instance."""

    def __init__(self, ttl_seconds: int, cooldown_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._cooldown_seconds = cooldown_seconds
        self._prices: dict[tuple[str, date], tuple[Any, float]] = {}
        self._rate_limited_until = 0.0
        self._lock = RLock()

    def get(self, ticker: str, start_date: date) -> Any | None:
        with self._lock:
            cached = self._prices.get((ticker, start_date))
            if cached is None:
                return None
            prices, expires_at = cached
            if time.monotonic() >= expires_at:
                del self._prices[(ticker, start_date)]
                return None
            return prices.copy()

    def set(self, ticker: str, start_date: date, prices: Any) -> None:
        with self._lock:
            self._prices[(ticker, start_date)] = (prices.copy(), time.monotonic() + self._ttl_seconds)

    def retry_after_seconds(self) -> int:
        with self._lock:
            return max(0, math.ceil(self._rate_limited_until - time.monotonic()))

    def activate_cooldown(self) -> int:
        with self._lock:
            self._rate_limited_until = time.monotonic() + self._cooldown_seconds
            return self._cooldown_seconds


market_data_cache = MarketDataCache(
    ttl_seconds=int(os.getenv("MARKET_DATA_CACHE_TTL_SECONDS", "900")),
    cooldown_seconds=int(os.getenv("YAHOO_RATE_LIMIT_COOLDOWN_SECONDS", "300")),
)


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

    cached_prices = market_data_cache.get(normalized_ticker, start_date)
    if cached_prices is not None:
        logger.info("Using cached Yahoo Finance prices for ticker=%s start_date=%s", normalized_ticker, start_date)
        return cached_prices

    retry_after_seconds = market_data_cache.retry_after_seconds()
    if retry_after_seconds:
        raise YahooRateLimitError(retry_after_seconds)

    try:
        import yfinance as yf
        from yfinance.exceptions import YFRateLimitError as YFinanceRateLimitError
    except Exception as exc:
        logger.exception("Yahoo Finance client could not be initialized")
        raise MarketDataProviderError(
            "Could not initialize the Yahoo Finance client. Verify deployment dependencies and retry shortly."
        ) from exc

    try:
        frame = yf.Ticker(normalized_ticker).history(
            start=start_date.isoformat(),
            auto_adjust=False,
            timeout=15,
            raise_errors=True,
        )
    except YFinanceRateLimitError as exc:
        cooldown_seconds = market_data_cache.activate_cooldown()
        logger.warning(
            "Yahoo Finance rate-limited ticker=%s; blocking new market-data calls for %s seconds",
            normalized_ticker,
            cooldown_seconds,
        )
        raise YahooRateLimitError(cooldown_seconds) from exc
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
    market_data_cache.set(normalized_ticker, start_date, close)
    return close
