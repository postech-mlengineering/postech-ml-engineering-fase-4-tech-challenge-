import time
import hashlib
import json

class PredictionCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache = {}  # key: (value, expire_time)

    def _generate_key(self, prices: list[float]) -> str:
        prices_str = json.dumps(prices)
        return hashlib.sha256(prices_str.encode('utf-8')).hexdigest()

    def get(self, prices: list[float]) -> float | None:
        key = self._generate_key(prices)
        if key in self.cache:
            val, expire = self.cache[key]
            if time.time() < expire:
                return val
            else:
                del self.cache[key]
        return None

    def set(self, prices: list[float], val: float):
        key = self._generate_key(prices)
        expire = time.time() + self.ttl
        self.cache[key] = (val, expire)

prediction_cache = PredictionCache()
