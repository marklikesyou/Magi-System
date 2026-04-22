"""Tests for magi.core.utils caching, retry, sanitization, and token tracking."""

from __future__ import annotations

import pytest

from magi.core.utils import (
    LRUCache,
    RateLimiter,
    TokenTracker,
    retry_with_backoff,
    sanitize_input,
)


def test_lru_cache_hit():
    """Cache returns the stored value for a previously inserted key."""
    cache = LRUCache(max_size=4)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_lru_cache_eviction():
    """The oldest (least-recently-used) entry is evicted when the cache is full."""
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_access_refreshes():
    """Accessing a key moves it to the end so it is not the next to be evicted."""
    cache = LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_retry_with_backoff_succeeds():
    """A function that fails once then succeeds is retried and returns the result."""
    call_count = {"n": 0}

    @retry_with_backoff(max_retries=3, initial_delay=0.0, exceptions=(ValueError,))
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert call_count["n"] == 2


def test_retry_with_backoff_exhausted():
    """When all retries fail the last exception is re-raised."""

    @retry_with_backoff(max_retries=2, initial_delay=0.0, exceptions=(RuntimeError,))
    def always_fails():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        always_fails()


def test_rate_limiter_waits_between_calls():
    now = {"t": 0.0}
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now["t"]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now["t"] += delay

    limiter = RateLimiter(60, clock=fake_clock, sleeper=fake_sleep)

    limiter.acquire()
    limiter.acquire()

    assert sleeps == [1.0]


def test_sanitize_input_strips_control_characters():
    """Control characters (except newline / tab) are removed."""
    dirty = "hello\x00world\x07!"
    clean = sanitize_input(dirty)
    assert "\x00" not in clean
    assert "\x07" not in clean
    assert "helloworld!" == clean


def test_sanitize_input_strips_script_tags():
    """Inline <script> blocks are removed."""
    text = 'before<script>alert("x")</script>after'
    assert "script" not in sanitize_input(text).lower()
    assert "before" in sanitize_input(text)
    assert "after" in sanitize_input(text)


def test_sanitize_input_truncates():
    """Input longer than max_length is truncated."""
    long_text = "a" * 500
    result = sanitize_input(long_text, max_length=100)
    assert len(result) <= 100


def test_token_tracker_accumulates():
    """Multiple track() calls accumulate input and output token counts."""
    tracker = TokenTracker()
    tracker.track("hello world", "response one", model="gpt-4o-mini")
    tracker.track("another prompt", "response two", model="gpt-4o-mini")

    stats = tracker.get_stats()
    assert stats["total_input_tokens"] > 0
    assert stats["total_output_tokens"] > 0
    assert (
        stats["total_tokens"]
        == stats["total_input_tokens"] + stats["total_output_tokens"]
    )


def test_token_tracker_reset():
    """reset() zeroes all counters."""
    tracker = TokenTracker()
    tracker.track("some input", "some output", model="gpt-4o-mini")
    tracker.reset()

    stats = tracker.get_stats()
    assert stats["total_input_tokens"] == 0
    assert stats["total_output_tokens"] == 0
    assert stats["total_tokens"] == 0
    assert stats["estimated_cost_usd"] == 0.0
