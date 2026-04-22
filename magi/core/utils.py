from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

T = TypeVar("T")

logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int = 100):
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


def hash_query(query: str, constraints: str = "") -> str:
    content = f"{query}||{constraints}"
    return hashlib.sha256(content.encode()).hexdigest()


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception: BaseException | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {e}"
                        )

            if last_exception is not None:
                raise last_exception
            raise RuntimeError(
                f"{func.__name__} failed without raising a tracked exception"
            )

        return wrapper

    return decorator


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.requests_per_minute = max(0, int(requests_per_minute))
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def acquire(self) -> None:
        if self.requests_per_minute <= 0:
            return
        min_interval = 60.0 / self.requests_per_minute
        with self._lock:
            now = self._clock()
            wait_for = max(0.0, self._next_allowed_at - now)
            if wait_for > 0.0:
                self._sleeper(wait_for)
                now = self._clock()
            self._next_allowed_at = max(now, self._next_allowed_at) + min_interval


def sanitize_input(text: str, max_length: int = 10000) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = text[:max_length]

    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)

    return text.strip()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    if TIKTOKEN_AVAILABLE:
        try:
            if "gpt" in model.lower():
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4
    else:
        return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    tokens = count_tokens(text, model)
    if tokens <= max_tokens:
        return text

    if TIKTOKEN_AVAILABLE:
        try:
            if "gpt" in model.lower():
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            encoded = encoding.encode(text)
            truncated = encoded[:max_tokens]
            return encoding.decode(truncated)
        except Exception:
            char_limit = (max_tokens * 4) - 100
            return text[:char_limit]
    else:
        char_limit = (max_tokens * 4) - 100
        return text[:char_limit]


class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.model_costs = {
            "gpt-4o": {"input": 0.03, "output": 0.06},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},
            "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},
        }

    def track(self, input_text: str, output_text: str, model: str):
        input_tokens = count_tokens(input_text, model)
        output_tokens = count_tokens(output_text, model)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if model in self.model_costs:
            costs = self.model_costs[model]
            self.total_cost += (input_tokens * costs["input"] / 1000) + (
                output_tokens * costs["output"] / 1000
            )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
        }

    def reset(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
