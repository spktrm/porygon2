"""JaxCacheNoiseFilter (rl/online/main.py): drops the harmless sub-2s
"PERSISTENT COMPILATION CACHE MISS" / "took < 2.00 seconds" pair that
JAX_EXPLAIN_CACHE_MISSES logs on every startup, while keeping rarer
"not writing" reasons visible — those are what EXPLAIN_CACHE_MISSES exists
to catch (see the filter's own docstring and .env's comment on it).
"""

import logging

from rl.online.main import JaxCacheNoiseFilter


def make_record(msg: str, *args) -> logging.LogRecord:
    return logging.LogRecord(
        name="jax._src.compiler",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_drops_the_miss_announcement():
    record = make_record(
        "PERSISTENT COMPILATION CACHE MISS for '%s' with key %r",
        "jit__normal",
        "somekey",
    )
    assert JaxCacheNoiseFilter().filter(record) is False


def test_drops_the_sub_2s_reason():
    record = make_record(
        "Not writing persistent cache entry for '%s' because it took < %.2f "
        "seconds to compile (%.2fs)",
        "jit_multiply",
        2.0,
        0.18,
    )
    assert JaxCacheNoiseFilter().filter(record) is False


def test_keeps_the_host_callbacks_reason():
    record = make_record(
        "Not writing persistent cache entry for '%s' because it uses host "
        "callbacks (e.g. from jax.debug.print or breakpoint)",
        "jit_something",
    )
    assert JaxCacheNoiseFilter().filter(record) is True


def test_keeps_the_process_id_reason():
    record = make_record("Not writing persistent cache entry since process_id != 0")
    assert JaxCacheNoiseFilter().filter(record) is True


def test_keeps_unrelated_compiler_messages():
    record = make_record("Persistent compilation cache hit for '%s'", "jit_stage")
    assert JaxCacheNoiseFilter().filter(record) is True
