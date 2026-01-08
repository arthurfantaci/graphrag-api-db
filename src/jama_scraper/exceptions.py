"""Custom exceptions for the Jama Guide Scraper.

Provides a hierarchy of exceptions for different error conditions:
- ScraperError: Base exception for all scraper errors
- FetchError: Error during content fetching
- PlaywrightNotAvailableError: Playwright package not installed
- BrowserNotInstalledError: Browser binaries not installed
"""


class ScraperError(Exception):
    """Base exception for scraper errors."""


class FetchError(ScraperError):
    """Error during content fetching.

    Attributes:
        url: The URL that failed to fetch.
    """

    def __init__(self, url: str, message: str) -> None:
        """Initialize FetchError.

        Args:
            url: The URL that failed to fetch.
            message: Description of what went wrong.
        """
        self.url = url
        super().__init__(f"Failed to fetch {url}: {message}")


class PlaywrightNotAvailableError(ScraperError):
    """Playwright package not installed.

    Raised when browser mode is requested but playwright is not available.
    """

    def __init__(self) -> None:
        """Initialize PlaywrightNotAvailableError."""
        super().__init__(
            "Playwright not installed. Install with: uv sync --group browser"
        )


class BrowserNotInstalledError(ScraperError):
    """Playwright browser binaries not installed.

    Raised when playwright is installed but browser binaries are missing.
    """

    def __init__(self) -> None:
        """Initialize BrowserNotInstalledError."""
        super().__init__("Browser not installed. Run: playwright install chromium")
