# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an async Python scraper that consolidates Jama Software's "Essential Guide to Requirements Management and Traceability" into LLM-friendly formats (JSON, JSONL, Markdown) for use with AI agents, MCP servers, and RAG systems.

## Commands

### Install and Run
```bash
# Using UV (recommended)
uv sync
uv run python run.py

# Using pip
pip install -e .
python run.py
```

### CLI Usage
```bash
graphrag-kg                           # Default: outputs JSON + JSONL
graphrag-kg -o ./data                 # Custom output directory
graphrag-kg -f json -f jsonl -f markdown  # Multiple formats
graphrag-kg --include-html            # Include raw HTML in output
graphrag-kg --browser                 # Use Playwright for JS-rendered content
```

### Browser Mode (Playwright)
For JavaScript-rendered content (e.g., YouTube embeds):
```bash
# Install browser dependencies
uv sync --group browser
playwright install chromium

# Run with browser mode (slower but captures dynamic content)
graphrag-kg --browser
```

### Development
```bash
uv sync --group dev          # Install with dev tools
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check src/         # Type check
```

## Code Style Standards

This project uses strict Python standards enforced by Ruff and ty:

- **Python 3.13** - Latest language features
- **Line length**: 88 characters (Black standard)
- **Docstrings**: Google-style, required for all public functions
- **Type annotations**: Required for all public functions
- **Import sorting**: isort via Ruff

### Ruff Rule Sets
- Core: E, W, F (pycodestyle, pyflakes)
- Quality: B, C4, UP, SIM, RUF (bugbear, comprehensions, pyupgrade)
- Docs: D (pydocstyle with Google convention)
- Security: S (bandit)
- Types: ANN, TCH (annotations, TYPE_CHECKING optimization)
- Structure: TRY, EM, PIE, PT, RET, ARG, PL (best practices)

### VS Code Setup
The `.vscode/` directory contains:
- `settings.json` - Ruff integration, Pylance, pytest, format-on-save
- `extensions.json` - Recommended extensions (Ruff, Pylance, etc.)
- `launch.json` - Debug configurations for CLI, run.py, and pytest

## Architecture

The scraper follows a pipeline architecture with pluggable fetching strategies:

```
┌─────────────────────────────────────────────────────────────┐
│                     JamaGuideScraper                        │
│  - Orchestrates scraping pipeline                           │
│  - Uses parser for HTML → Markdown conversion               │
│  - Delegates fetching to Fetcher abstraction                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Fetcher (Protocol)                        │
│  async def fetch(url: str) -> str | None                    │
│  async context manager support                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│     HttpxFetcher      │   │   PlaywrightFetcher   │
│  - Fast (~50ms/page)  │   │  - Slow (~2s/page)    │
│  - Static HTML only   │   │  - Full JS rendering  │
│  - Default            │   │  - --browser flag     │
└───────────────────────┘   └───────────────────────┘
```

### Core Modules

1. **config.py** - URL configurations, chapter/article mappings, rate limiting settings. `ChapterConfig` and `ArticleConfig` dataclasses define the guide structure. Some chapters have incomplete article lists and are discovered dynamically.

2. **fetcher.py** - Protocol-based fetcher abstraction (PEP 544 structural subtyping):
   - `Fetcher` - Protocol defining the interface with async context manager support
   - `FetcherConfig` - Frozen dataclass for configuration (rate limit, concurrency, timeout)
   - `HttpxFetcher` - Fast HTTP client for static HTML (default)
   - `PlaywrightFetcher` - Headless browser for JS-rendered content (lazy init)
   - `create_fetcher()` - Factory function for instantiation

3. **scraper.py** - `JamaGuideScraper` orchestrates the pipeline:
   - `scrape_all()` → `_discover_all_articles()` → `_scrape_all_chapters()` → `_scrape_glossary()`
   - Uses fetcher abstraction via dependency injection

4. **parser.py** - `HTMLParser` converts HTML to Markdown and extracts metadata:
   - `parse_article()` - Extracts title, markdown, sections, cross-references, images, videos
   - `parse_glossary()` - Handles multiple glossary HTML patterns
   - `discover_articles()` - Finds article links from chapter overview pages

5. **models.py** - Pydantic models with computed fields for word/character counts. Includes `VideoReference` for embedded media tracking.

6. **exceptions.py** - Custom exception hierarchy:
   - `ScraperError` - Base exception
   - `FetchError` - Content fetching failures
   - `PlaywrightNotAvailableError` - Package not installed
   - `BrowserNotInstalledError` - Browser binaries not installed

## Key Design Decisions

- **Protocol Pattern (PEP 544)** - Structural subtyping for fetcher abstraction enables testing and extensibility
- **Strategy Pattern** - Swappable fetching strategies (httpx vs Playwright) via `create_fetcher()` factory
- **Dependency Injection** - Scraper receives fetcher, doesn't create it internally
- **Lazy Initialization** - Playwright browser only started on first request
- **Frozen Dataclass** - `FetcherConfig` is immutable and hashable
- Async with semaphore concurrency (default 3 parallel requests) for respectful scraping
- Exponential backoff retry on HTTP errors (max 3 retries)
- Articles auto-discovered from chapter overviews when not pre-configured
- Cross-references tracked with internal/external classification for knowledge graphs
- JSONL format provides self-contained records per article for easy RAG chunking
