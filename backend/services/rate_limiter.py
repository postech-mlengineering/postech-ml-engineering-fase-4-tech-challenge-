import time

class IPRateLimiter:
    def __init__(self, limit: int = 5, window_seconds: int = 60):
        self.limit = limit
        self.window = window_seconds
        self.requests = {}  # ip: list of timestamps

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Keep only requests within the active window
        self.requests[ip] = [ts for ts in self.requests[ip] if now - ts < self.window]
        
        if len(self.requests[ip]) < self.limit:
            self.requests[ip].append(now)
            return True
        
        return False

# Prediction endpoint rate limiter (e.g., limit 5 requests per minute)
predict_rate_limiter = IPRateLimiter(limit=5, window_seconds=60)
