"""Simple token-bucket rate limiter for outbound HTTP requests.

Used by the contest scraper to stay well under LeetCode's rate limits.
Single-threaded; not safe for concurrent use.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    """Token-bucket rate limiter.

    Example:
        limiter = RateLimiter(rps=0.25)  # 1 request every 4 seconds
        for url in urls:
            limiter.wait()
            requests.get(url)
    """

    def __init__(
        self,
        rps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rps <= 0:
            raise ValueError(f"rps must be positive, got {rps}")
        self._min_interval: float = 1.0 / rps
        self._clock = clock
        self._sleep = sleep
        self._next_ready_at: float = clock()

    def wait(self) -> None:
        """Block until the next request is allowed."""
        now = self._clock()
        delay = self._next_ready_at - now
        if delay > 0:
            self._sleep(delay)
            now = self._clock()
        self._next_ready_at = max(now, self._next_ready_at) + self._min_interval

    def penalize(self, extra_seconds: float) -> None:
        """Push the next-ready time out by `extra_seconds` after a 429 or 5xx."""
        if extra_seconds < 0:
            raise ValueError(f"extra_seconds must be non-negative, got {extra_seconds}")
        self._next_ready_at = max(self._next_ready_at, self._clock()) + extra_seconds
