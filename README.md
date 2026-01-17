# Jama Requirements Management Guide Scraper

A Python ETL pipeline that scrapes Jama Software's **"The Essential Guide to Requirements Management and Traceability"** and loads it into a Neo4j knowledge graph using the `neo4j_graphrag` library for GraphRAG retrieval.

## Features

- **Async scraping** with `httpx` for efficient parallel fetching
- **Neo4j GraphRAG integration** using `neo4j_graphrag.SimpleKGPipeline`
- **LangChain HTML chunking** with `HTMLHeaderTextSplitter` for semantic document splitting
- **Schema-constrained entity extraction** with 10 node types and 10 relationship types
- **Industry taxonomy normalization** consolidating 100+ variants into 18 canonical industries
- **Glossary-to-concept linking** with fuzzy matching via `rapidfuzz`
- **Post-extraction validation** with comprehensive quality checks

## Architecture

```
                   ┌──────────────────────────────────────────────┐
                   │              JamaGuideScraper                │
                   │  - Async HTTP fetching (httpx)               │
                   │  - HTML → Markdown conversion                │
                   │  - Cross-reference extraction                │
                   └───────────────────┬──────────────────────────┘
                                       │ RequirementsManagementGuide
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          neo4j_graphrag Pipeline                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────────────┐    │
│  │ JamaHTMLLoader  │───▶│ HTMLSplitter    │───▶│ LLMEntityRelExtractor │    │
│  │ (DataLoader)    │    │ (LangChain)     │    │ (Schema-constrained)  │    │
│  └─────────────────┘    └─────────────────┘    └───────────────────────┘    │
│                                                            │                 │
│                                                            ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────────────────┐    │
│  │ IndustryNormal. │◀───│ EntityNormalizer│◀───│ Neo4jWriter           │    │
│  │ (18 industries) │    │ (deduplication) │    │ (Knowledge Graph)     │    │
│  └─────────────────┘    └─────────────────┘    └───────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────────────┐
                   │               Neo4j Database                  │
                   │  - Article, Chapter, Chunk nodes             │
                   │  - Concept, Industry, Standard entities      │
                   │  - Vector embeddings for semantic search     │
                   └──────────────────────────────────────────────┘
```

## Requirements

- **Python 3.13+**
- **Neo4j 5.x** with APOC plugin
- **OpenAI API key** for embeddings and entity extraction
- **UV** (recommended) or pip for package management

## Installation

### Using UV (Recommended)

```bash
# Clone or copy the project
cd jama-guide-scraper

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

Create a `.env` file with the following variables:

```bash
# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# OpenAI API
OPENAI_API_KEY=sk-your-api-key

# Optional: Model configuration
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

## Usage

### Quick Start

```bash
# Run the full pipeline: scrape → extract → load to Neo4j
jama-scrape

# With validation report
jama-scrape --validate

# Skip resource node creation (images, videos, webinars)
jama-scrape --skip-resources
```

### Programmatic Usage

```python
import asyncio
from jama_scraper import run_scraper

# Run the full ETL pipeline
async def main():
    guide = await run_scraper(
        validate=True,
        skip_resources=False,
    )
    print(f"Loaded {guide.total_articles} articles to Neo4j")

asyncio.run(main())
```

### Running Validation Only

```python
from neo4j import GraphDatabase
from jama_scraper.validation.queries import run_all_validations

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
results = asyncio.run(run_all_validations(driver))
print(f"Validation passed: {results['validation_passed']}")
```

## Knowledge Graph Schema

### Node Types

| Label | Description | Key Properties |
|-------|-------------|----------------|
| `Article` | Source document | `article_id`, `title`, `url` |
| `Chapter` | Document grouping | `chapter_number`, `title` |
| `Chunk` | Text fragment for RAG | `text`, `embedding` |
| `Concept` | Domain concept | `name`, `description` |
| `Industry` | Business sector | `name`, `regulated` |
| `Standard` | Compliance standard | `name`, `organization` |
| `Tool` | Software/methodology | `name`, `vendor` |
| `Challenge` | Problem domain | `name`, `severity` |
| `BestPractice` | Recommended approach | `name`, `category` |
| `ProcessStage` | Lifecycle phase | `name`, `sequence` |

### Relationship Types

- `ADDRESSES` - Challenge → Solution
- `REQUIRES` - Dependency relationships
- `COMPONENT_OF` - Part-whole relationships
- `RELATED_TO` - General associations
- `APPLIES_TO` - Standard → Industry
- `USED_BY` - Tool → Role
- `PRODUCES` - Process → Artifact
- `DEFINES` - Standard → Concept
- `FROM_ARTICLE` - Chunk → Article
- `MENTIONED_IN` - Entity → Chunk

## Project Structure

```
jama-guide-scraper/
├── src/jama_scraper/
│   ├── __init__.py           # Package exports
│   ├── cli.py                # Command-line interface
│   ├── scraper.py            # Async web scraper
│   ├── parser.py             # HTML → Markdown parser
│   ├── models_core.py        # Pydantic models
│   ├── models/               # Resource models
│   │   └── resource.py       # Image, Video, Webinar
│   ├── chunking/             # LangChain text splitting
│   │   ├── config.py         # HierarchicalChunkingConfig
│   │   ├── hierarchical_chunker.py
│   │   └── adapter.py        # LangChain adapter
│   ├── extraction/           # Entity extraction
│   │   ├── schema.py         # NODE_TYPES, RELATIONSHIPS
│   │   ├── prompts.py        # LLM extraction prompts
│   │   └── pipeline.py       # SimpleKGPipeline factory
│   ├── loaders/              # Data loading
│   │   ├── html_loader.py    # JamaHTMLLoader
│   │   └── index_builder.py  # Article index utilities
│   ├── postprocessing/       # Entity normalization
│   │   ├── industry_taxonomy.py  # 18 canonical industries
│   │   ├── normalizer.py     # Entity deduplication
│   │   └── glossary_linker.py
│   ├── graph/                # Neo4j supplementary
│   │   ├── supplementary.py  # Chapter/Resource nodes
│   │   └── constraints.py    # Indexes and constraints
│   └── validation/           # Quality checks
│       ├── queries.py        # Validation Cypher
│       └── reporter.py       # Report generation
├── tests/                    # Comprehensive test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_models.py
│   ├── test_chunking.py
│   ├── test_extraction.py
│   ├── test_loaders.py
│   ├── test_postprocessing.py
│   └── test_validation.py
├── pyproject.toml            # Project configuration
├── test_query.py             # Knowledge graph query demo
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
```

## Querying the Knowledge Graph

### Test Script (Recommended)

A ready-to-use test script is included for evaluating the knowledge graph:

```bash
# Run with default query
uv run python test_query.py

# Run with custom query
uv run python test_query.py "What is impact analysis?"
```

The script demonstrates four query approaches:
1. **Vector similarity search** - Semantic matching on chunk embeddings
2. **Chunk-to-entity traversal** - Find entities mentioned in retrieved chunks
3. **Direct entity search** - Search entities by name pattern
4. **Relationship exploration** - Show connections for a specific entity

Example output:
```
╭────────────────────── Jama Guide Knowledge Graph Test ───────────────────────╮
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

This tool is for educational and research purposes. Please respect Jama Software's terms of service and use the scraped content responsibly. The content remains the intellectual property of Jama Software.

## License

MIT License - See LICENSE file for details.
