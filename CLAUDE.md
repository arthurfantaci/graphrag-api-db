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

### Pre-Ingestion Validation
```bash
# Validate TOC discovery against live site (~2 seconds, no API keys needed)
uv run python examples/validate_toc_discovery.py
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
│  │ GuideScraper → Fetcher (httpx/Playwright) → HTML Parser            │  │
│  │ Output: RequirementsManagementGuide (JSON/JSONL)                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 2: EXTRACT & EMBED (neo4j_graphrag SimpleKGPipeline)             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ GuideHTMLLoader → HierarchicalHTMLSplitter → LLMEntityRelExtractor│  │
│  │ - Optional Chonkie SemanticChunker (Savitzky-Golay boundaries)    │  │
│  │ - Schema-constrained entity extraction (10 node types)             │  │
│  │ - Voyage AI voyage-4 embeddings (auto-detected from VOYAGE_API_KEY)│  │
│  │ - Fallback: OpenAI text-embedding-3-small if no Voyage key        │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 3: NORMALIZE (Entity Post-Processing)                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 1. EntityNormalizer.normalize_all_entities() — lowercase + trim   │  │
│  │ 2. EntityNormalizer.deduplicate_by_name() — merge duplicates      │  │
│  │ 3. EntityCleanupNormalizer.run_cleanup() — generics + plurals     │  │
│  │ 4. IndustryNormalizer.consolidate_industries() — 100+ → 18       │  │
│  │ 5. MentionedInBackfiller.backfill() — MENTIONED_IN + APPLIES_TO  │  │
│  │ 6. EntitySummarizer.summarize() — LLM entity descriptions        │  │
│  │ 7. LangExtractAugmenter.augment() — source grounding (optional) │  │
│  │ 8. CommunityDetector.detect_communities() — Leiden clustering   │  │
│  │ 9. CommunitySummarizer.summarize_communities() — LLM summaries  │  │
│  │ 10. CommunityEmbedder.embed_community_summaries() — vectors   │  │
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

1. **config.py** - URL constants (`BASE_URL`, `GLOSSARY_URL`), rate limiting settings. `ChapterConfig` and `ArticleConfig` dataclasses define the guide structure (populated dynamically from the TOC).

2. **fetcher.py** - Protocol-based fetcher abstraction (PEP 544 structural subtyping):
   - `Fetcher` - Protocol defining the interface with async context manager support
   - `FetcherConfig` - Frozen dataclass for configuration (rate limit, concurrency, timeout)
   - `HttpxFetcher` - Fast HTTP client for static HTML (default)
   - `PlaywrightFetcher` - Headless browser for JS-rendered content (lazy init)
   - `create_fetcher()` - Factory function for instantiation

3. **scraper.py** - `GuideScraper` orchestrates scraping, `run_scraper()` runs full 5-stage pipeline:
   - `scrape_all()` → `_discover_guide_structure()` → `_scrape_all_chapters()` → `_scrape_glossary()` → `_enrich_webinar_thumbnails()`
   - Dynamic discovery from `#chapter-menu` TOC (single HTTP request replaces per-chapter discovery)
   - Uses fetcher abstraction via dependency injection

4. **parser.py** - `HTMLParser` converts HTML to Markdown and extracts metadata:
   - `parse_article()` - Extracts title, markdown, sections, cross-references, images, videos
   - `parse_glossary()` - Handles multiple glossary HTML patterns
   - `parse_chapter_menu()` - Parses `#chapter-menu` TOC for dynamic chapter/article discovery
   - `extract_og_image()` - Extracts `<meta property="og:image">` URL from HTML (used for webinar thumbnail fallback)

5. **models/content.py** - Pydantic models with computed fields for word/character counts.

6. **exceptions.py** - Custom exception hierarchy including `Neo4jConfigError`.

**Extraction (Stage 2):**

7. **extraction/schema.py** - Knowledge graph schema:
   - `NODE_TYPES` - 10 types: Concept, Challenge, Artifact, Industry, Standard, Tool, etc.
   - `RELATIONSHIP_TYPES` - 10 types: ADDRESSES, REQUIRES, COMPONENT_OF, APPLIES_TO, etc.
   - `PATTERNS` - ~30 validation patterns for entity names

8. **extraction/pipeline.py** - `neo4j_graphrag` integration:
   - `KGPipelineConfig` - Pipeline configuration dataclass
   - `create_kg_pipeline()` - Factory for SimpleKGPipeline
   - `process_guide_with_pipeline()` - Main processing function

9. **loaders/html_loader.py** - `GuideHTMLLoader` implements neo4j_graphrag DataLoader interface.

10. **chunking/hierarchical_chunker.py** - LangChain-based hierarchical HTML splitting with optional Chonkie semantic chunking (stage 2 fallback to RCTS).

11. **embeddings/voyage.py** - `VoyageAIEmbeddings` implementing `neo4j_graphrag.embeddings.base.Embedder` for Voyage AI voyage-4 asymmetric embeddings.

**Post-Processing (Stage 3):**

11. **postprocessing/normalizer.py** - `EntityNormalizer`: name normalization (lowercase, trim) and duplicate merging.

12. **postprocessing/entity_cleanup.py** - `EntityCleanupNormalizer`: generic entity deletion, plural-to-singular merging.

13. **postprocessing/industry_taxonomy.py** - `IndustryNormalizer` consolidates 100+ variants to 18 canonical industries.

14. **postprocessing/mentioned_in_backfill.py** - `MentionedInBackfiller`: creates MENTIONED_IN and APPLIES_TO relationships.

15. **postprocessing/entity_summarizer.py** - `EntitySummarizer`: LLM-generated entity descriptions.

16. **postprocessing/langextract_augmenter.py** - `LangExtractAugmenter`: post-extraction entity augmentation with source grounding (text span provenance).

**Graph Algorithms (Stage 3, steps 8-10):**

17. **graph/community_detection.py** - `CommunityDetector`: Leiden community detection using `leidenalg` + `igraph` on semantic edges only.

18. **graph/community_summarizer.py** - `CommunitySummarizer`: LLM-generated community summaries (gpt-4o-mini), creates Community nodes linked via IN_COMMUNITY.

19. **graph/community_embedder.py** - `CommunityEmbedder`: Voyage AI voyage-4 embeddings for Community node summaries, enables vector search over communities.

**Graph Building (Stage 4):**

20. **graph/supplementary.py** - `SupplementaryGraphBuilder`:
    - Chapter structure, Resource nodes, Glossary-to-concept linking

21. **graph/constraints.py** - `ConstraintManager` for Neo4j indexes and constraints. Also provides `create_community_vector_index()` for `community_summary_embeddings` vector index (1024d cosine).

**Validation (Stage 5):**

22. **validation/queries.py** - `ValidationQueries` class with Cypher checks.

23. **validation/fixes.py** - `ValidationFixer` for data repair operations.

24. **validation/reporter.py** - `ValidationReporter` for report generation.

## Key Design Decisions

**Neo4j Constraint Compatibility:**
- `ExtractionGleaner._merge_gleaned_results()` MERGE pattern must include `__Entity__` label and set `__KGBuilder__` on creation — without these, gleaned entities are invisible to entity resolver and dedup
- neo4j_graphrag 1.13.0+ uses `CREATE` + `apoc.create.addLabels()`, NOT `MERGE` for entities
- Entity-type uniqueness constraints (Concept.name, Challenge.name, etc.) MUST NOT exist — they cause silent batch rollbacks via IndexEntryConflictException
- Only structural constraints are safe: Article, Chunk, Chapter, Image, Video, Webinar, Definition
- Entity deduplication is handled by neo4j_graphrag's entity resolution step

**Pipeline Runtime:**
- Full re-ingestion: ~1.5 hours (101 articles, ~1 article/min)
- No built-in graph clear command — use `MATCH (n) DETACH DELETE n` manually
- Rich progress bars don't flush to redirected output; monitor via Neo4j node count queries
- Direct OpenAI API calls (gleaning) need `response_format={"type": "json_object"}`
- Gleaning runs 2 passes by default (each pass queries Neo4j for latest state)

**Scraping Layer:**
- **Protocol Pattern (PEP 544)** - Structural subtyping for fetcher abstraction enables testing and extensibility
- **Strategy Pattern** - Swappable fetching strategies (httpx vs Playwright) via `create_fetcher()` factory
- **Dependency Injection** - Scraper receives fetcher, doesn't create it internally
- **Lazy Initialization** - Playwright browser only started on first request
- Async with semaphore concurrency (default 3 parallel requests) for respectful scraping
- Exponential backoff retry on HTTP errors (max 3 retries)
- **Dynamic TOC Discovery** - All chapters/articles discovered from `#chapter-menu` on the guide's main page (single HTTP request, no static fallback)
- **OG Image Enrichment** - `_enrich_webinar_thumbnails()` runs inside `scrape_all()` after guide construction, fetches unique webinar landing pages to backfill null `thumbnail_url` via `<meta property="og:image">`
- Uses `html.parser` (not lxml) for TOC parsing because the source HTML has `<div>` inside `<ul>` — lxml restructures this invalid nesting
- Cloudflare blocks default httpx User-Agent — always set `User-Agent: GuideScraper/0.1.0 (Educational/Research)` for direct requests
- Content models (`WebinarReference`, etc.) are mutable Pydantic v2 `BaseModel` — direct attribute assignment works for post-scrape enrichment

**Knowledge Graph Layer:**
- **Schema-Constrained Extraction** - 10 node types and 10 relationship types prevent schema drift
- **Industry Taxonomy** - 100+ variants normalized to 18 canonical industries
- **Entity Deduplication** - Plural forms merged (e.g., "requirement" + "requirements" → "requirement")
- **Hierarchical Chunking** - LangChain HTMLHeaderTextSplitter preserves document structure; optional Chonkie SemanticChunker for stage 2
- **Voyage AI Embeddings** - Auto-detected from `VOYAGE_API_KEY` env var; asymmetric `input_type` ("document" for indexing, "query" for search)
- **Community Detection** - Leiden algorithm via `leidenalg` (reference implementation). Only semantic relationship types projected (not structural). Communities get LLM-generated summaries.
- **LangExtract Augmentation** - Post-processing entity augmentation with source grounding (text span provenance). Runs after all other cleanup to avoid introducing duplicates.
- **Supplementary Structure** - Chapter, Resource, and Glossary nodes add navigational context

**Data Quality:**
- **Validation Framework** - Comprehensive checks for orphans, duplicates, and invalid patterns
- **Repair Operations** - Safe fixes with dry-run preview mode
- **Report Archiving** - `ValidationReport.save()` auto-archives existing reports with ISO 8601 timestamps (e.g. `validation_report_2026-02-20T001437.md`) before writing the new one
- **Vector Embeddings** - Voyage AI voyage-4 (preferred) or OpenAI text-embedding-3-small for semantic search
- **Fix ordering** - delete degenerate → re-index → chunk_ids → webinar titles → relabel → backfill MENTIONED_IN → definitions → generic → plurals
- **Pass/fail checks** - orphan_chunks, duplicates, chunk_ids, chunk_index, plural_duplicates (industry count advisory)
