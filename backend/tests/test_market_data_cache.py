import sys
import unittest
from datetime import date

sys.path.insert(0, "backend")

from services.training.data_service import MarketDataCache


class CopyablePrices:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return CopyablePrices(self.value)


class MarketDataCacheTests(unittest.TestCase):
    def test_returns_cached_copy(self):
        cache = MarketDataCache(ttl_seconds=60, cooldown_seconds=300)
        prices = CopyablePrices("prices")
        cache.set("AAPL", date(2025, 1, 1), prices)
        cached = cache.get("AAPL", date(2025, 1, 1))
        self.assertEqual(cached.value, "prices")
        self.assertIsNot(cached, prices)

    def test_cooldown_reports_retry_after(self):
        cache = MarketDataCache(ttl_seconds=60, cooldown_seconds=30)
        self.assertEqual(cache.activate_cooldown(), 30)
        self.assertGreater(cache.retry_after_seconds(), 0)


if __name__ == "__main__":
    unittest.main()
