"""Tests for the token-bucket rate limiter.

Uses injected fake ``clock`` and ``sleep`` callables so the tests are
deterministic and run in microseconds.
"""

from __future__ import annotations

import pytest

from lrs.data._rate_limiter import RateLimiter


class FakeClock:
    """Monotonic fake clock that advances only when ``sleep`` is called."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, "sleep called with negative duration"
        self.sleeps.append(seconds)
        self.now += seconds


def test_rejects_non_positive_rps():
    with pytest.raises(ValueError):
        RateLimiter(rps=0)
    with pytest.raises(ValueError):
        RateLimiter(rps=-1)


def test_first_call_does_not_sleep():
    fake = FakeClock()
    limiter = RateLimiter(rps=1.0, clock=fake.clock, sleep=fake.sleep)

    limiter.wait()

    assert fake.sleeps == []


def test_enforces_minimum_interval_between_calls():
    fake = FakeClock()
    limiter = RateLimiter(rps=2.0, clock=fake.clock, sleep=fake.sleep)  # 0.5s interval

    limiter.wait()  # first call: no sleep
    limiter.wait()  # second call: should sleep 0.5s

    assert fake.sleeps == [pytest.approx(0.5)]


def test_does_not_sleep_when_caller_is_already_slow():
    fake = FakeClock()
    limiter = RateLimiter(rps=1.0, clock=fake.clock, sleep=fake.sleep)

    limiter.wait()
    fake.now += 5.0  # caller did 5 seconds of work between requests
    limiter.wait()

    assert fake.sleeps == [], "should not sleep when natural gap exceeds interval"


def test_penalize_extends_the_next_ready_time():
    fake = FakeClock()
    limiter = RateLimiter(rps=10.0, clock=fake.clock, sleep=fake.sleep)

    limiter.wait()  # t=0
    limiter.penalize(3.0)  # next ready at >=3s
    limiter.wait()

    assert fake.sleeps, "penalize should force a sleep on the next wait()"
    assert sum(fake.sleeps) >= 3.0


def test_penalize_rejects_negative_duration():
    limiter = RateLimiter(rps=1.0)
    with pytest.raises(ValueError):
        limiter.penalize(-0.1)
