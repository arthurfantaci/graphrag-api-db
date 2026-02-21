"""Content fetcher implementations using Protocol-based abstraction.

This module provides pluggable fetching strategies:
- HttpxFetcher: Fast, lightweight HTTP client (default)
- PlaywrightFetcher: Headless browser for JavaScript-rendered content

Example:
    async with HttpxFetcher() as fetcher:
        html = await fetcher.fetch("https://example.com")

    # Or using the factory function:
    async with create_fetcher(use_browser=True) as fetcher:
        html = await fetcher.fetch("https://example.com")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    RATE_LIMIT_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)
from .exceptions import (
    BrowserNotInstalledError,
    PlaywrightNotAvailableError,
)

if TYPE_CHECKING:
    from types import TracebackType

# HTTP status codes
HTTP_NOT_FOUND = 404

console = Console()


@dataclass(frozen=True)
class FetcherConfig:
    """Configuration for content fetchers.

    Attributes:
        rate_limit_delay: Minimum seconds between requests.
        max_concurrent: Maximum parallel requests.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on failure.
        user_agent: User-Agent header for requests.
    """

    rate_limit_delay: float = RATE_LIMIT_DELAY_SECONDS
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
    timeout: float = REQUEST_TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    user_agent: str = "GuideScraper/0.1.0 (Educational/Research)"


@runtime_checkable
class Fetcher(Protocol):
    """Protocol defining the fetcher interface.

    Fetchers must implement async context manager protocol for
    proper resource management.
    """

    async def fetch(self, url: str) -> str | None:
        """Fetch content from URL.

        Args:
            url: The URL to fetch.

        Returns:
            HTML content as string, or None if fetch failed (e.g., 404).
        """
        ...

    async def close(self) -> None:
        """Release resources."""
        ...

    async def __aenter__(self) -> Fetcher:
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        ...


class HttpxFetcher:
    """Fast HTTP fetcher using httpx for static HTML content.

    Features:
    - Async HTTP/2 support
    - Automatic redirect following
    - Semaphore-based concurrency control
    - Rate limiting between requests
    - Exponential backoff retry on errors

    Example:
        async with HttpxFetcher() as fetcher:
            html = await fetcher.fetch("https://example.com")
    """

    def __init__(self, config: FetcherConfig | None = None) -> None:
        """Initialize HttpxFetcher.

        Args:
            config: Optional fetcher configuration. Uses defaults if not provided.
        """
        self._config = config or FetcherConfig()
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._last_request_time = 0.0

    async def __aenter__(self) -> HttpxFetcher:
        """Initialize HTTP client on context entry."""
        self._client = httpx.AsyncClient(
            timeout=self._config.timeout,
            follow_redirects=True,
            headers={"User-Agent": self._config.user_agent},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close HTTP client on context exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> str | None:
        """Fetch URL with rate limiting and retry logic.

        Args:
            url: The URL to fetch.

        Returns:
            HTML content as string, or None if fetch failed (e.g., 404).

        Raises:
            RuntimeError: If fetcher not initialized (not used as context manager).
        """
        if not self._client:
            msg = "Fetcher not initialized. Use as async context manager."
            raise RuntimeError(msg)
        return await self._fetch_with_retry(url)

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _fetch_with_retry(self, url: str) -> str | None:
        """Internal fetch with retry decorator.

        Args:
            url: The URL to fetch.

        Returns:
            HTML content as string, or None for 404 responses.
        """
        async with self._semaphore:
            await self._apply_rate_limit()
            try:
                response = await self._client.get(url)  # type: ignore[union-attr]
                response.raise_for_status()
                self._last_request_time = time.monotonic()
                return response.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == HTTP_NOT_FOUND:
                    console.print(f"[yellow]404 Not Found: {url}[/]")
                    return None
                raise
            except Exception as e:
                console.print(f"[red]Error fetching {url}: {e}[/]")
                raise

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._config.rate_limit_delay:
            await asyncio.sleep(self._config.rate_limit_delay - elapsed)


class PlaywrightFetcher:
    """Headless browser fetcher for JavaScript-rendered content.

    Features:
    - Full JavaScript execution
    - Waits for network idle before returning content
    - Lazy browser initialization
    - Context isolation per request
    - Automatic resource cleanup

    Example:
        async with PlaywrightFetcher() as fetcher:
            html = await fetcher.fetch("https://example.com")
    """

    def __init__(self, config: FetcherConfig | None = None) -> None:
        """Initialize PlaywrightFetcher.

        Args:
            config: Optional fetcher configuration. Uses defaults if not provided.
        """
        self._config = config or FetcherConfig()
        self._playwright: object | None = None
        self._browser: object | None = None
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._last_request_time = 0.0

    async def __aenter__(self) -> PlaywrightFetcher:
        """Initialize browser on context entry."""
        await self._ensure_browser()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close browser on context exit."""
        await self.close()

    async def close(self) -> None:
        """Close the browser and release resources."""
        if self._browser:
            await self._browser.close()  # type: ignore[union-attr]
            self._browser = None
        if self._playwright:
            await self._playwright.stop()  # type: ignore[union-attr]
            self._playwright = None

    async def _ensure_browser(self) -> None:
        """Lazily initialize browser on first use.

        Raises:
            PlaywrightNotAvailableError: If playwright package not installed.
            BrowserNotInstalledError: If browser binaries not installed.
        """
        if self._browser is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise PlaywrightNotAvailableError() from e

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(  # type: ignore[union-attr]
                headless=True
            )
            console.print("[cyan]Browser initialized (Playwright/Chromium)[/]")
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                raise BrowserNotInstalledError() from e
            raise

    async def fetch(self, url: str) -> str | None:
        """Fetch URL using headless browser.

        Creates a new browser context for each request to ensure isolation.

        Args:
            url: The URL to fetch.

        Returns:
            HTML content as string, or None if fetch failed.
        """
        await self._ensure_browser()

        async with self._semaphore:
            await self._apply_rate_limit()

            context = await self._browser.new_context(  # type: ignore[union-attr]
                user_agent=self._config.user_agent
            )
            page = await context.new_page()

            try:
                await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=self._config.timeout * 1000,  # Playwright uses ms
                )
                content = await page.content()
                self._last_request_time = time.monotonic()
                return content
            except Exception as e:
                console.print(f"[red]Error fetching {url}: {e}[/]")
                return None
            finally:
                await context.close()

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._config.rate_limit_delay:
            await asyncio.sleep(self._config.rate_limit_delay - elapsed)


def create_fetcher(
    use_browser: bool = False,
    config: FetcherConfig | None = None,
) -> HttpxFetcher | PlaywrightFetcher:
    """Factory function to create appropriate fetcher.

    Args:
        use_browser: If True, use Playwright for JS rendering.
        config: Optional fetcher configuration.

    Returns:
        Configured fetcher instance (HttpxFetcher or PlaywrightFetcher).

    Example:
        async with create_fetcher(use_browser=True) as fetcher:
            html = await fetcher.fetch(url)
    """
    if use_browser:
        return PlaywrightFetcher(config)
    return HttpxFetcher(config)
