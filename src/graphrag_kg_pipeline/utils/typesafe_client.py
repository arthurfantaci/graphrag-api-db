"""Factory for the TypeSafe System One client.

The TypeSafe SDK retries rate-limit (429), overload (529), and transient
5xx responses on its own through ``RetryPolicy``. Like the Voyage AI client,
it must NOT be wrapped in the tenacity decorators from ``utils/retry.py``;
doing so would multiply the retry count.
"""

from __future__ import annotations

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from graphrag_kg_pipeline.exceptions import TypeSafeConfigError

DEFAULT_TYPESAFE_MODEL = "jev-latest"
"""Model alias that tracks the newest Jev release."""

TYPESAFE_RETRY_POLICY = RetryPolicy(max_retries=3)
"""Three retries after the first attempt, matching ``openai_retry``'s 3 attempts.

Backoff, jitter, and ``Retry-After`` handling keep the SDK defaults.
"""


def create_typesafe_client(
    api_key: str,
    model: str = DEFAULT_TYPESAFE_MODEL,
) -> AsyncTypeSafeClient:
    """Create an async TypeSafe client with the project retry policy.

    Args:
        api_key: TypeSafe API key. Usually ``KGPipelineConfig.typesafe_api_key``.
        model: System One model name or alias.

    Returns:
        Configured ``AsyncTypeSafeClient``. Close it with ``await client.aclose()``
        or use it as an async context manager.

    Raises:
        TypeSafeConfigError: If ``api_key`` is empty or whitespace.
    """
    if not api_key or not api_key.strip():
        raise TypeSafeConfigError

    return AsyncTypeSafeClient(
        api_key=api_key.strip(),
        model=model,
        retry=TYPESAFE_RETRY_POLICY,
    )
