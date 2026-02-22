"""Shared retry decorators for LLM API calls.

Provides tenacity-based retry decorators for the 5 post-processing call sites
that bypass neo4j_graphrag's built-in rate limit handling. Stage 2 extraction
(SimpleKGPipeline) already has its own retry via OpenAILLM.DEFAULT_RATE_LIMIT_HANDLER.

Note: Voyage AI's client has built-in retry (``max_retries`` parameter) and
should NOT use these decorators to avoid double-retry cascades.
"""

from __future__ import annotations

import logging

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# tenacity's before_sleep_log requires a stdlib logger, NOT structlog.
# See: https://tenacity.readthedocs.io/en/latest/#before-and-after-retry
_tenacity_logger = logging.getLogger("graphrag_kg_pipeline.retry")

openai_retry = retry(
    retry=retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )
    ),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, min=1, max=60),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
)
"""Retry decorator for direct ``AsyncOpenAI`` calls (gleaning, summarizers).

Matches neo4j_graphrag's built-in ``RetryRateLimitHandler`` parameters:
``multiplier=2``, 3 attempts, random exponential backoff capped at 60s.
"""


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if an exception is a wrapped or direct rate limit error.

    langextract wraps OpenAI errors in ``InferenceRuntimeError(original=e)``,
    so tenacity's ``retry_if_exception_type`` would never match. This predicate
    unwraps the ``.original`` attribute to check the underlying cause.

    Args:
        exc: The exception to inspect.

    Returns:
        True if the exception (or its wrapped original) is a retryable API error.
    """
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    original = getattr(exc, "original", None)
    return isinstance(
        original, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    )


langextract_retry = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, min=1, max=60),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
)
"""Retry decorator for ``langextract.extract()`` calls.

Uses a custom predicate because langextract wraps OpenAI errors in
``InferenceRuntimeError``. The ``lx.extract()`` function is synchronous,
so this decorator is sync-compatible.
"""
