"""Tiny in-process TTL cache for live-data endpoints (mirrors the old st.cache_data ttl)."""

import time
from typing import Callable, TypeVar

T = TypeVar("T")

_store: dict[str, tuple[float, object]] = {}


def cached(key: str, ttl_seconds: float, fn: Callable[[], T]) -> T:
    """Return a cached value for `key` if still fresh, else recompute via `fn`."""
    now = time.time()
    hit = _store.get(key)
    if hit is not None and now - hit[0] < ttl_seconds:
        return hit[1]  # type: ignore[return-value]
    value = fn()
    _store[key] = (now, value)
    return value
