# GraphRAG Knowledge Graph Pipeline

[![CI](https://github.com/arthurfantaci/graphrag-api-db/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurfantaci/graphrag-api-db/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg?logo=ruff)](https://docs.astral.sh/ruff/)

A Python ETL pipeline that scrapes Jama Software's **"The Essential Guide to Requirements Management and Traceability"** and loads it into a Neo4j knowledge graph using the `neo4j_graphrag` library for GraphRAG retrieval.

## Features

- **Dynamic guide discovery** from live TOC — no static configuration required
- **Async scraping** with `httpx` (default) or Playwright (browser mode for JS-rendered content)
- **Neo4j GraphRAG integration** using `neo4j_graphrag.SimpleKGPipeline`
- **Schema-constrained entity extraction** with 10 node types and 10 relationship types
- **Gleaning** — 2-pass LLM extraction catches 20-30% additional entities
- **Hierarchical HTML chunking** with LangChain `HTMLHeaderTextSplitter`; optional Chonkie semantic chunking
- **Voyage AI voyage-4 embeddings** (1024d, asymmetric); auto-detected from `VOYAGE_API_KEY` with OpenAI fallback
- **10-step entity post-processing** — normalization, deduplication, cleanup, industry taxonomy (100+ → 18), backfill, LLM summaries, source grounding, community detection, community summaries, community embeddings
- **Leiden community detection** with LLM-generated community summaries and vector embeddings
- **Cross-label entity deduplication** — merges same-name entities with different type labels
- **Glossary-to-concept linking** with fuzzy matching via `rapidfuzz`
- **Supplementary graph structure** — Chapter, Resource (Image/Video/Webinar), and Glossary nodes
- **Validation framework** with comprehensive quality checks, repair operations, and auto-archived reports
- **Preflight validation** — checks Neo4j connectivity, APOC availability, and vector index dimensions
- **Cost estimation** via `--dry-run` flag

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        run_scraper() Pipeline                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: SCRAPE                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ GuideScraper → Fetcher (httpx/Playwright) → HTML Parser            │  │
│  │ - Dynamic TOC discovery from #chapter-menu (single HTTP request)   │  │
│  │ - OG image enrichment for webinar thumbnails                       │  │
│  │ Output: RequirementsManagementGuide (JSON/JSONL)                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 2: EXTRACT & EMBED (neo4j_graphrag SimpleKGPipeline)             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ GuideHTMLLoader → HierarchicalHTMLSplitter → LLMEntityRelExtractor│  │
│  │ - Gleaning: 2-pass LLM extraction for improved recall              │  │
│  │ - Schema-constrained entity extraction (10 node types)             │  │
│  │ - Voyage AI voyage-4 embeddings (auto-detected from VOYAGE_API_KEY)│  │
│  │ - Fallback: OpenAI text-embedding-3-small if no Voyage key        │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 3: NORMALIZE (Entity Post-Processing — 10 steps)                 │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  1. EntityNormalizer — lowercase + trim                            │  │
│  │  2. EntityNormalizer — deduplicate by name                         │  │
│  │  3. EntityCleanupNormalizer — generics + plurals                   │  │
│  │  4. IndustryNormalizer — consolidate 100+ → 18 industries         │  │
│  │  5. MentionedInBackfiller — MENTIONED_IN + APPLIES_TO rels        │  │
│  │  6. EntitySummarizer — LLM entity descriptions                    │  │
│  │  7. LangExtractAugmenter — source grounding (text provenance)    │  │
│  │  8. CommunityDetector — Leiden clustering                         │  │
│  │  9. CommunitySummarizer — LLM community summaries                │  │
│  │ 10. CommunityEmbedder — Voyage AI community vectors             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 4: SUPPLEMENT (Graph Structure)                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ SupplementaryGraphBuilder                                          │  │
│  │ - Chapter nodes + Article relationships                            │  │
│  │ - Resource nodes (Image, Video, Webinar)                           │  │
│  │ - Glossary structure + concept linking                             │  │
│  │ - Community summary vector index                                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                           │
│  Stage 5: VALIDATE (Optional)                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ValidationQueries → ValidationReporter                             │  │
│  │ - Orphan chunk detection, entity quality, relationship patterns    │  │
│  │ - Auto-archived reports with ISO 8601 timestamps                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Requirements

- **Python 3.13+**
- **Neo4j 5.x** with APOC plugin
- **OpenAI API key** for entity extraction, gleaning, and summaries
- **Voyage AI API key** (optional, preferred) for embeddings — falls back to OpenAI
- **UV** (recommended) or pip for package management

## Installation

### Using UV (Recommended)

```bash
# Clone or copy the project
cd graphrag-kg-pipeline

# Install dependencies
uv sync

# Copy environment template and configure
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

Create a `.env` file (see `.env.example` for full template):

```bash
# OpenAI (required) — LLM for extraction, gleaning, summaries
OPENAI_API_KEY=sk-your-api-key

# Voyage AI (optional — preferred for embeddings)
# When set, voyage-4 (1024d, asymmetric) is auto-detected.
# When absent, falls back to OpenAI text-embedding-3-small.
VOYAGE_API_KEY=pa-your-api-key

# Neo4j connection
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

## Usage

### Quick Start

```bash
# Run the full pipeline: scrape → extract → load to Neo4j
graphrag-kg

# Explicit scrape subcommand (same as above)
graphrag-kg scrape

# With validation report after pipeline completes
graphrag-kg scrape --validate

# Skip resource node creation (images, videos, webinars)
graphrag-kg scrape --skip-resources

# Skip all supplementary structure (chapters, resources, glossary)
graphrag-kg scrape --skip-supplementary

# Scrape only - skip Neo4j processing (saves JSON/JSONL only)
graphrag-kg scrape --scrape-only

# Estimate costs without running (dry run)
graphrag-kg scrape --dry-run

# Use headless browser for JavaScript-rendered content
graphrag-kg scrape --browser

# Validate the graph and generate report
graphrag-kg validate

# Save validation report to file
graphrag-kg validate -o validation_report.md

# Preview what fixes would do without applying
graphrag-kg validate --fix --dry-run

# Apply all fixes (chunk_ids + entity cleanup)
graphrag-kg validate --fix

# Apply only specific fix types
graphrag-kg validate --fix-chunk-ids   # Safe, additive operation
graphrag-kg validate --fix-entities    # Merges plurals, deletes generic entities
```

### Pre-Ingestion Validation

```bash
# Validate TOC discovery against live site (~2 seconds, no API keys needed)
uv run python examples/validate_toc_discovery.py
```

### Programmatic Usage

```python
import asyncio
from graphrag_kg_pipeline import run_scraper

# Run the full Neo4j GraphRAG pipeline
async def main():
    guide = await run_scraper(
        output_dir="./output",      # Intermediate files directory
        run_validation=True,        # Run validation after loading
        skip_resources=False,       # Include images, videos, webinars
        skip_supplementary=False,   # Include chapters, glossary structure
    )
    print(f"Loaded {guide.total_articles} articles to Neo4j")

asyncio.run(main())

# Scrape only (no Neo4j processing)
async def scrape_only():
    guide = await run_scraper(scrape_only=True)
    print(f"Scraped {guide.total_articles} articles to JSON/JSONL")

asyncio.run(scrape_only())
```

### Running Validation Only

```python
import asyncio
from neo4j import AsyncGraphDatabase
from graphrag_kg_pipeline.validation import generate_validation_report

async def validate_graph():
    driver = AsyncGraphDatabase.driver(
        "neo4j+s://xxx.databases.neo4j.io",
        auth=("neo4j", "password")
    )
    try:
        report = await generate_validation_report(driver, "neo4j")
        print(report.to_markdown())
        print(f"Validation passed: {report.validation_passed}")
    finally:
        await driver.close()

asyncio.run(validate_graph())
```

## Knowledge Graph Schema

### Node Types

| Label | Description | Key Properties |
|-------|-------------|----------------|
| `Article` | Source document | `article_id`, `title`, `url` |
| `Chapter` | Document grouping | `chapter_number`, `title` |
| `Chunk` | Text fragment for RAG | `text`, `embedding`, `index` |
| `Community` | Leiden community cluster | `communityId`, `summary`, `summary_embedding` |
| `Concept` | Domain concept | `name`, `definition`, `aliases` |
| `Industry` | Business sector | `name` |
| `Standard` | Compliance standard | `name` |
| `Tool` | Software/methodology | `name` |
| `Challenge` | Problem domain | `name` |
| `BestPractice` | Recommended approach | `name` |
| `ProcessStage` | Lifecycle phase | `name` |
| `Role` | Job function | `name` |
| `Methodology` | Process framework | `name` |
| `Artifact` | Work product | `name` |
| `Image` | Visual resource | `src`, `alt_text` |
| `Video` | Video resource | `url`, `title` |
| `Webinar` | Webinar resource | `url`, `title`, `thumbnail_url` |
| `Definition` | Glossary entry | `term`, `definition`, `acronym` |

### Relationship Types

**Semantic (LLM-extracted):**
- `ADDRESSES` - Concept/BestPractice/Tool → Challenge
- `REQUIRES` - Dependency relationships
- `COMPONENT_OF` - Part-whole relationships
- `RELATED_TO` - General semantic associations
- `ALTERNATIVE_TO` - Competing approaches
- `USED_BY` - Tool/Artifact/Concept → Role/Industry
- `APPLIES_TO` - Standard → Industry
- `PRODUCES` - Process → Artifact
- `DEFINES` - Standard → Concept
- `PREREQUISITE_FOR` - Sequential dependencies

**Structural (pipeline-created):**
- `FROM_ARTICLE` - Chunk → Article
- `MENTIONED_IN` - Entity → Chunk
- `IN_COMMUNITY` - Entity → Community
- `HAS_ARTICLE` - Chapter → Article
- `CONTAINS_DEFINITION` - Glossary → Definition
- `DEFINES_CONCEPT` - Definition → Concept

## Project Structure

```
graphrag-kg-pipeline/
├── src/graphrag_kg_pipeline/
│   ├── __init__.py           # Package exports
│   ├── cli.py                # Command-line interface
│   ├── config.py             # URL configs, rate limiting settings
│   ├── fetcher.py            # Protocol-based fetcher (httpx/Playwright)
│   ├── scraper.py            # Async web scraper + pipeline orchestration
│   ├── parser.py             # HTML → Markdown parser + TOC discovery
│   ├── preflight.py          # Pre-ingestion validation
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── models/               # Pydantic data models
│   │   ├── content.py        # Article, Chapter, Glossary
│   │   └── resource.py       # Image, Video, Webinar
│   ├── chunking/             # LangChain text splitting
│   │   ├── config.py         # HierarchicalChunkingConfig
│   │   ├── hierarchical_chunker.py
│   │   └── adapter.py        # LangChain → neo4j_graphrag adapter
│   ├── embeddings/           # Custom embedding providers
│   │   └── voyage.py         # VoyageAIEmbeddings (Embedder interface)
│   ├── extraction/           # Entity extraction
│   │   ├── schema.py         # NODE_TYPES, RELATIONSHIP_TYPES, PATTERNS
│   │   ├── prompts.py        # LLM extraction prompts
│   │   ├── pipeline.py       # SimpleKGPipeline factory
│   │   └── gleaning.py       # Multi-pass extraction refinement
│   ├── loaders/              # Data loading
│   │   ├── html_loader.py    # GuideHTMLLoader (DataLoader interface)
│   │   └── index_builder.py  # Article index utilities
│   ├── postprocessing/       # Entity normalization (6 modules)
│   │   ├── normalizer.py     # Name normalization + deduplication
│   │   ├── entity_cleanup.py # Plural/generic entity handling
│   │   ├── industry_taxonomy.py  # 100+ → 18 canonical industries
│   │   ├── mentioned_in_backfill.py  # MENTIONED_IN + APPLIES_TO
│   │   ├── entity_summarizer.py      # LLM entity descriptions
│   │   ├── langextract_augmenter.py  # Source grounding (text provenance)
│   │   └── glossary_linker.py        # Glossary → Concept linking
│   ├── graph/                # Graph algorithms + structure (5 modules)
│   │   ├── community_detection.py   # Leiden clustering (leidenalg + igraph)
│   │   ├── community_summarizer.py  # LLM community summaries
│   │   ├── community_embedder.py    # Voyage AI community embeddings
│   │   ├── supplementary.py  # Chapter/Resource/Glossary nodes
│   │   └── constraints.py    # Indexes, constraints, vector indexes
│   └── validation/           # Quality checks
│       ├── queries.py        # Validation Cypher queries
│       ├── fixes.py          # Data repair utilities
│       └── reporter.py       # Report generation + auto-archive
├── tests/                    # Comprehensive test suite (232 tests)
│   ├── conftest.py           # Pytest fixtures
│   ├── test_models.py
│   ├── test_chunking.py
│   ├── test_extraction.py
│   ├── test_loaders.py
│   ├── test_postprocessing.py
│   ├── test_community.py
│   ├── test_langextract.py
│   ├── test_embeddings.py
│   ├── test_preflight.py
│   ├── test_scraper.py
│   ├── test_smoke.py
│   └── test_validation.py
├── examples/                 # Usage demonstrations
│   ├── query_knowledge_graph.py      # 4 query approaches demo
│   ├── validate_toc_discovery.py     # Pre-ingestion TOC validation
│   ├── backfill_entity_labels.py     # Entity label repair utility
│   └── diagnose_concept_anomaly.py   # Debugging: concept count analysis
├── pyproject.toml            # Project configuration
├── .env.example              # Environment template
└── CLAUDE.md                 # AI assistant guidance
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_extraction.py
```

### Linting & Formatting

```bash
uv run ruff check .          # Lint check
uv run ruff check . --fix    # Auto-fix issues
uv run ruff format .         # Format code
```

### Type Checking

```bash
uv run ty check src/
```

## Validation Queries

After loading, validate the graph quality:

```cypher
// Check for orphan chunks (should be 0)
MATCH (c:Chunk) WHERE NOT (c)-[:FROM_ARTICLE]->() RETURN count(c)

// Check industry count (should be ≤18)
MATCH (i:Industry) RETURN count(i)

// Check for duplicate entities
MATCH (n) WHERE n:Concept OR n:Industry
WITH n.name AS name, labels(n)[0] AS label, count(n) AS cnt
WHERE cnt > 1 RETURN label, name, cnt

// Check embedding coverage
MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c)

// Check community summary embedding coverage
MATCH (c:Community) WHERE c.summary_embedding IS NULL RETURN count(c)
```

## Querying the Knowledge Graph

### Example Script (Recommended)

A ready-to-use example script is included for evaluating the knowledge graph:

```bash
# Run with default query
uv run python examples/query_knowledge_graph.py

# Run with custom query
uv run python examples/query_knowledge_graph.py "What is impact analysis?"

# Custom entity search term
uv run python examples/query_knowledge_graph.py "What is impact analysis?" --search "impact"
```

The script demonstrates four query approaches:
1. **Vector similarity search** - Semantic matching on chunk embeddings
2. **Chunk-to-entity traversal** - Find entities mentioned in retrieved chunks
3. **Direct entity search** - Search entities by name pattern
4. **Relationship exploration** - Show connections for a specific entity

Example output:
```
╭────────────────────── Guide Knowledge Graph Test ────────────────────────────╮
│ Query: What can you tell me about Requirements Tracing?                      │
╰──────────────────────────────────────────────────────────────────────────────╯

1. Vector Similarity Search (semantic match)
------------------------------------------------------------
Result 1 (score: 0.851)
In simple terms, requirements traceability is the process of creating
and maintaining connections between different development artifacts...

3. Direct Entity Search (name contains 'trac')
------------------------------------------------------------
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Type     ┃ Name                     ┃ Connections ┃ Definition               ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Concept  │ Traceability             │         370 │ the only way to know...  │
│ Concept  │ Requirements Traceability│         147 │ the practice of linking..│
│ Concept  │ Live Traceability        │          84 │ The ability for any...   │
└──────────┴──────────────────────────┴─────────────┴──────────────────────────┘
```

### Semantic Search (RAG)

```cypher
// Find relevant chunks using vector similarity
CALL db.index.vector.queryNodes('chunk_embedding', 5, $query_embedding)
YIELD node, score
MATCH (node)-[:FROM_ARTICLE]->(a:Article)
RETURN a.title, node.text, score
ORDER BY score DESC
```

### Community Search

```cypher
// Find relevant communities using vector similarity
CALL db.index.vector.queryNodes('community_summary_embeddings', 3, $query_embedding)
YIELD node, score
RETURN node.communityId, node.summary, score
ORDER BY score DESC
```

### Knowledge Graph Traversal

```cypher
// Find concepts related to a specific industry
MATCH (i:Industry {name: "automotive"})<-[:APPLIES_TO]-(s:Standard)
MATCH (s)-[:DEFINES]->(c:Concept)
RETURN DISTINCT c.name, s.name
```

### Article Context

```cypher
// Get all entities mentioned in an article
MATCH (a:Article {article_id: "ch1-art1"})<-[:FROM_ARTICLE]-(c:Chunk)
MATCH (c)<-[:MENTIONED_IN]-(e)
RETURN labels(e)[0] AS type, e.name, count(*) AS mentions
ORDER BY mentions DESC
```

## Legal Notice

The scraped content remains the intellectual property of Jama Software. Please respect their terms of service when using this pipeline. This project demonstrates a production-grade GraphRAG pipeline architecture and is not affiliated with Jama Software.

## License

MIT License - See LICENSE file for details.
