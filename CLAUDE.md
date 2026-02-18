# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a complete Neo4j GraphRAG pipeline that scrapes Jama Software's "Essential Guide to Requirements Management and Traceability" and loads it into a Neo4j knowledge graph using `neo4j_graphrag`'s SimpleKGPipeline. The pipeline performs LLM-based entity extraction, industry normalization, and vector embeddings for semantic RAG retrieval.

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
# Run full Neo4j GraphRAG pipeline
graphrag-kg                           # Default: scrape + Neo4j pipeline
graphrag-kg scrape                    # Explicit scrape subcommand
graphrag-kg scrape -o ./data          # Custom output directory
graphrag-kg scrape --validate         # Run validation after pipeline
graphrag-kg scrape --scrape-only      # Scrape only, no Neo4j processing
graphrag-kg scrape --skip-resources   # Skip Image/Video/Webinar nodes
graphrag-kg scrape --skip-supplementary  # Skip chapters, resources, glossary
graphrag-kg scrape --dry-run          # Estimate costs without running
graphrag-kg scrape --browser          # Use Playwright for JS-rendered content

# Validate and fix data quality
graphrag-kg validate                  # Run validation checks
graphrag-kg validate -o report.md     # Save report to file
graphrag-kg validate --fix --dry-run  # Preview fixes without applying
graphrag-kg validate --fix            # Apply all fixes
graphrag-kg validate --fix-chunk-ids  # Fix only chunk IDs (safe)
graphrag-kg validate --fix-entities   # Fix only entity quality
```

### Browser Mode (Playwright)
For JavaScript-rendered content (e.g., YouTube embeds):
```bash
# Install browser dependencies
uv sync --group browser
playwright install chromium

# Run with browser mode (slower but captures dynamic content)
graphrag-kg scrape --browser
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
- **Line length**: 100 characters
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

The pipeline executes 5 stages to transform web content into a queryable knowledge graph:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        run_scraper() Pipeline                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: SCRAPE                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ JamaGuideScraper → Fetcher (httpx/Playwright) → HTML Parser       │  │
│  │ Output: RequirementsManagementGuide (JSON/JSONL)                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 2: EXTRACT & EMBED (neo4j_graphrag SimpleKGPipeline)             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ JamaHTMLLoader → HierarchicalHTMLSplitter → LLMEntityRelExtractor │  │
│  │ - Schema-constrained entity extraction (10 node types)             │  │
│  │ - OpenAI embeddings for vector search                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 3: NORMALIZE (Entity Post-Processing)                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ IndustryNormalizer (18 canonical) → EntityNormalizer (dedupe)      │  │
│  │ - Plural entity merging (requirement vs requirements)              │  │
│  │ - Generic entity cleanup                                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 4: SUPPLEMENT (Graph Structure)                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ SupplementaryGraphBuilder                                          │  │
│  │ - Chapter nodes + Article relationships                            │  │
│  │ - Resource nodes (Image, Video, Webinar)                           │  │
│  │ - Glossary structure + concept linking                             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 5: VALIDATE (Optional)                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ValidationQueries → ValidationReporter                             │  │
│  │ - Orphan chunk detection, entity quality, relationship patterns    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Fetcher Abstraction (Stage 1 Detail)

```
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

**Scraping (Stage 1):**

1. **config.py** - URL configurations, chapter/article mappings, rate limiting settings. `ChapterConfig` and `ArticleConfig` dataclasses define the guide structure.

2. **fetcher.py** - Protocol-based fetcher abstraction (PEP 544 structural subtyping):
   - `Fetcher` - Protocol defining the interface with async context manager support
   - `FetcherConfig` - Frozen dataclass for configuration (rate limit, concurrency, timeout)
   - `HttpxFetcher` - Fast HTTP client for static HTML (default)
   - `PlaywrightFetcher` - Headless browser for JS-rendered content (lazy init)
   - `create_fetcher()` - Factory function for instantiation

3. **scraper.py** - `JamaGuideScraper` orchestrates scraping, `run_scraper()` runs full 5-stage pipeline:
   - `scrape_all()` → `_discover_all_articles()` → `_scrape_all_chapters()` → `_scrape_glossary()`
   - Uses fetcher abstraction via dependency injection

4. **parser.py** - `HTMLParser` converts HTML to Markdown and extracts metadata:
   - `parse_article()` - Extracts title, markdown, sections, cross-references, images, videos
   - `parse_glossary()` - Handles multiple glossary HTML patterns
   - `discover_articles()` - Finds article links from chapter overview pages

5. **models_core.py** - Pydantic models with computed fields for word/character counts.

6. **exceptions.py** - Custom exception hierarchy including `Neo4jConfigError`.

**Extraction (Stage 2):**

7. **extraction/schema.py** - Knowledge graph schema:
   - `NODE_TYPES` - 10 types: Concept, Challenge, Artifact, Industry, Standard, Tool, etc.
   - `RELATIONSHIP_TYPES` - 10 types: ADDRESSES, REQUIRES, COMPONENT_OF, APPLIES_TO, etc.
   - `PATTERNS` - ~30 validation patterns for entity names

8. **extraction/pipeline.py** - `neo4j_graphrag` integration:
   - `JamaKGPipelineConfig` - Pipeline configuration dataclass
   - `create_jama_kg_pipeline()` - Factory for SimpleKGPipeline
   - `process_guide_with_pipeline()` - Main processing function

9. **loaders/html_loader.py** - `JamaHTMLLoader` implements neo4j_graphrag DataLoader interface.

10. **chunking/hierarchical_chunker.py** - LangChain-based semantic HTML splitting.

**Post-Processing (Stage 3):**

11. **postprocessing/industry_taxonomy.py** - `IndustryNormalizer` consolidates 100+ variants to 18 canonical industries.

12. **postprocessing/entity_cleanup.py** - Plural/singular deduplication, generic entity removal.

13. **postprocessing/normalizer.py** - `EntityNormalizer` for entity deduplication.

**Graph Building (Stage 4):**

14. **graph/supplementary.py** - `SupplementaryGraphBuilder`:
    - Chapter structure, Resource nodes, Glossary-to-concept linking

15. **graph/constraints.py** - `ConstraintManager` for Neo4j indexes and constraints.

**Validation (Stage 5):**

16. **validation/queries.py** - `ValidationQueries` class with Cypher checks.

17. **validation/fixes.py** - `ValidationFixer` for data repair operations.

18. **validation/reporter.py** - `ValidationReporter` for report generation.

## Key Design Decisions

**Scraping Layer:**
- **Protocol Pattern (PEP 544)** - Structural subtyping for fetcher abstraction enables testing and extensibility
- **Strategy Pattern** - Swappable fetching strategies (httpx vs Playwright) via `create_fetcher()` factory
- **Dependency Injection** - Scraper receives fetcher, doesn't create it internally
- **Lazy Initialization** - Playwright browser only started on first request
- Async with semaphore concurrency (default 3 parallel requests) for respectful scraping
- Exponential backoff retry on HTTP errors (max 3 retries)
- Articles auto-discovered from chapter overviews when not pre-configured

**Knowledge Graph Layer:**
- **Schema-Constrained Extraction** - 10 node types and 10 relationship types prevent schema drift
- **Industry Taxonomy** - 100+ variants normalized to 18 canonical industries
- **Entity Deduplication** - Plural forms merged (e.g., "requirement" + "requirements" → "requirement")
- **Hierarchical Chunking** - LangChain HTMLHeaderTextSplitter preserves document structure
- **Supplementary Structure** - Chapter, Resource, and Glossary nodes add navigational context

**Data Quality:**
- **Validation Framework** - Comprehensive checks for orphans, duplicates, and invalid patterns
- **Repair Operations** - Safe fixes with dry-run preview mode
- **Vector Embeddings** - OpenAI text-embedding-3-small for semantic search
