# Claude Code Planning Document: Jama Guide ETL Pipeline Refactoring

## Executive Summary

This document provides comprehensive requirements for a **full rebuild** of the `jama-guide-scraper` ETL pipeline. The refactoring:

1. **Migrates** from LangExtract to `neo4j_graphrag`'s `SimpleKGPipeline` for schema-constrained entity extraction
2. **Implements** hierarchical 3-tier chunking with content storage
3. **Adds** a flexible Resource node pattern for images, webinars, videos, links, and definitions
4. **Fixes** data quality issues: case-variant duplicates, industry fragmentation, miscategorized entities
5. **Produces** a new Neo4j schema (MCP server updates will follow in a separate phase)

---

## Table of Contents

1. [Reference Materials](#1-reference-materials)
2. [Target Architecture](#2-target-architecture)
3. [Neo4j Schema Specification](#3-neo4j-schema-specification)
4. [Data Quality Requirements](#4-data-quality-requirements)
5. [Module Specifications](#5-module-specifications)
6. [neo4j_graphrag Integration](#6-neo4j_graphrag-integration)
   - 6.1-6.3: Schema, Pipeline, Supplementary Cypher
   - 6.4: Custom Extraction Prompts with Few-Shot Examples
   - 6.5: Entity Property Assignment
   - 6.6: LexicalGraphConfig for Domain-Specific Structure
7. [LangChain HTML Integration](#7-langchain-html-integration) **[NEW]**
   - 7.1: LangChain HTML Splitters Overview
   - 7.2: HTMLHeaderTextSplitter for Jama Content
   - 7.3: HTMLSectionSplitter with XSLT for Avia Theme
   - 7.4: LangChainTextSplitterAdapter for neo4j_graphrag
   - 7.5: Custom HTML DataLoader for Jama Content
   - 7.6: Complete Pipeline Assembly
8. [Hierarchical Chunking Strategy](#8-hierarchical-chunking-strategy)
9. [Resource Extraction](#9-resource-extraction)
10. [Industry Taxonomy](#10-industry-taxonomy)
11. [Implementation Phases](#11-implementation-phases)
12. [Testing Requirements](#12-testing-requirements)
13. [Validation Queries](#13-validation-queries)
14. [Appendices](#14-appendices)
    - A: Environment Variables
    - B: CLI Usage Examples
    - C: Schema Migration Notes
    - D: Sub-Agent Analysis Summary **[NEW]**

---

## 1. Reference Materials

### 1.1 Source Repositories

| Repository | Location | Purpose |
|------------|----------|---------|
| jama-guide-scraper | `/Users/arthurfantaci/Projects/jama-guide-scraper` | Current pipeline (to refactor) |
| genai-graphrag-python | `/Users/arthurfantaci/Projects/genai-graphrag-python` | Reference patterns for neo4j_graphrag |

### 1.2 Key Reference Files

```
# Current pipeline modules to understand
jama-guide-scraper/src/jama_scraper/
├── scraper.py              # KEEP: Async scraping with httpx
├── parser.py               # KEEP: HTML parsing, extend for resources
├── models.py               # REFACTOR: Add Resource models
├── extractor.py            # REPLACE: Migrate to neo4j_graphrag
├── extraction_schemas.py   # REPLACE: Convert to neo4j_graphrag schema
├── chunker.py              # REFACTOR: Hierarchical 3-tier
├── embedder.py             # KEEP: OpenAI ada-002 embeddings
├── graph_export.py         # REPLACE: Use neo4j_graphrag writer
└── neo4j_loader.py         # REPLACE: Use neo4j_graphrag writer

# neo4j_graphrag reference implementation
genai-graphrag-python/genai-graphrag-python/
└── kg_structured_builder.py  # Pattern for schema + supplementary Cypher
```

### 1.3 Target Neo4j Database

- **Database ID:** 234ea307
- **Instance:** Neo4j AuraDB
- **Current state:** Will be overwritten by refactored pipeline

---

## 2. Target Architecture

### 2.1 Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         REFACTORED ETL PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: SCRAPING (Keep existing, extend for resources)                     │
│  ═══════════════════════════════════════════════════════                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │   Scraper   │───▶│   Parser    │───▶│    Resource Extractor [NEW]     │  │
│  │  (httpx)    │    │ (HTML→MD)   │    │  - Images (content only)        │  │
│  │             │    │             │    │  - Webinars                      │  │
│  └─────────────┘    └─────────────┘    │  - Videos                        │  │
│                                        │  - External links                │  │
│                                        │  - Definition blocks             │  │
│                                        │  - Related article links         │  │
│                                        └─────────────────────────────────┘  │
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 2: CHUNKING [REFACTORED]                                              │
│  ══════════════════════════════                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    HIERARCHICAL CHUNKER                                 ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │ Level 0: Article Summary                                        │   ││
│  │  │ - One per article, ~300 tokens                                  │   ││
│  │  │ - Uses LLM-generated summary or first N chars                   │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                              │                                          ││
│  │                              ▼                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │ Level 1: Section Chunks                                         │   ││
│  │  │ - Natural heading boundaries (h2/h3)                            │   ││
│  │  │ - 500-1500 tokens each                                          │   ││
│  │  │ - [:CHILD_OF] → Level 0                                         │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                              │                                          ││
│  │                              ▼                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────┐   ││
│  │  │ Level 2: Sliding Window Chunks (for large sections)             │   ││
│  │  │ - 512 tokens with 64-token overlap                              │   ││
│  │  │ - [:CHILD_OF] → Level 1 parent                                  │   ││
│  │  │ - Only created when section > 1500 tokens                       │   ││
│  │  └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │  CRITICAL: Store actual text content in `text` property                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 3: ENTITY EXTRACTION [NEW - neo4j_graphrag]                           │
│  ═════════════════════════════════════════════════                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                 SimpleKGPipeline (schema-constrained)                   ││
│  │                                                                         ││
│  │  Schema Definition:                                                     ││
│  │  ├── NODE_TYPES: Concept, Challenge, Industry, Standard, Tool, etc.    ││
│  │  ├── RELATIONSHIP_TYPES: ADDRESSES, REQUIRES, APPLIES_TO, etc.         ││
│  │  └── PATTERNS: Valid (source, rel, target) triples                     ││
│  │                                                                         ││
│  │  Processing:                                                            ││
│  │  ├── LLM extracts entities constrained by schema                       ││
│  │  ├── Entity resolution merges duplicates                               ││
│  │  └── Normalization: toLower(trim(name)) + display_name                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 4: POST-PROCESSING [NEW]                                              │
│  ══════════════════════════════                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ ││
│  │  │ Industry        │  │ GlossaryTerm    │  │ Entity Type             │ ││
│  │  │ Taxonomy        │  │ Linking         │  │ Validation              │ ││
│  │  │ Mapping         │  │                 │  │                         │ ││
│  │  │ (96 → 19)       │  │ Auto-link       │  │ Reclassify              │ ││
│  │  │                 │  │ definitions     │  │ mistyped                │ ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 5: EMBEDDING                                                          │
│  ══════════════════                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  OpenAI text-embedding-ada-002 (1536 dimensions)                        ││
│  │  - Embed chunk.text content                                             ││
│  │  - Batch processing with checkpointing                                  ││
│  │  - Store in chunk.embedding property                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 6: GRAPH WRITING                                                      │
│  ══════════════════════                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  neo4j_graphrag GraphWriter + Supplementary Cypher                      ││
│  │  - MERGE with normalized names (no duplicates)                          ││
│  │  - Batch transactions for performance                                   ││
│  │  - Create indexes and constraints                                       ││
│  │  - Supplementary Cypher for Chapter/Article structure                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                         │                    │
│                                                         ▼                    │
│  PHASE 7: VALIDATION [NEW]                                                   │
│  ═════════════════════════                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Post-load validation queries                                           ││
│  │  - Check for case-variant duplicates (should be 0)                      ││
│  │  - Verify industry consolidation (should be ≤19)                        ││
│  │  - Check orphaned nodes                                                 ││
│  │  - Verify relationship counts                                           ││
│  │  - Validate property types (no accidental LISTs)                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

```toml
# pyproject.toml - Updated dependencies

[project]
name = "jama-guide-scraper"
version = "2.0.0"
description = "ETL pipeline for Jama Requirements Management Guide → Neo4j GraphRAG"
requires-python = ">=3.11"

dependencies = [
    # === KEEP: Existing working components ===
    "httpx>=0.27",              # Async HTTP client
    "beautifulsoup4>=4.12",     # HTML parsing
    "pydantic>=2.0",            # Data validation
    "rich>=13.0",               # CLI output
    "tiktoken>=0.5",            # Token counting
    "openai>=1.0",              # Embeddings
    "python-dotenv>=1.0",       # Environment variables
    
    # === NEW: neo4j_graphrag ecosystem ===
    "neo4j>=5.0",               # Neo4j driver
    "neo4j-graphrag>=1.0",      # KG builder pipeline
    
    # === NEW: Data quality ===
    "rapidfuzz>=3.0",           # Fuzzy matching for entity resolution
    
    # === NEW: Reliability ===
    "structlog>=24.0",          # Structured logging
    "tenacity>=8.0",            # Retry logic
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.10",
]
```

---

## 3. Neo4j Schema Specification

### 3.1 Node Labels and Properties

```cypher
// ═══════════════════════════════════════════════════════════════════════════
// CONTENT NODES
// ═══════════════════════════════════════════════════════════════════════════

// Chapter: Top-level content organization
(:Chapter {
    chapter_number: INTEGER,     // Primary key, unique constraint
    title: STRING,
    url: STRING
})

// Article: Individual content pages
(:Article {
    article_id: STRING,          // Primary key, e.g., "ch1-art1"
    title: STRING,
    url: STRING,
    summary: STRING,             // LLM-generated or extracted
    word_count: INTEGER,
    chapter_number: INTEGER      // Denormalized for query convenience
})

// GlossaryTerm: Terminology definitions
(:GlossaryTerm {
    term: STRING,                // Primary key, NORMALIZED: toLower(trim())
    display_term: STRING,        // Original casing for UI
    definition: STRING
})

// ═══════════════════════════════════════════════════════════════════════════
// CHUNK NODES (Hierarchical)
// ═══════════════════════════════════════════════════════════════════════════

(:Chunk {
    id: STRING,                  // Primary key, e.g., "ch1-art1-L0" or "ch1-art1-L1-s2"
    text: STRING,                // ACTUAL CONTENT - critical for RAG
    level: INTEGER,              // 0=article summary, 1=section, 2=sliding window
    chunk_type: STRING,          // "summary" | "section" | "window"
    heading: STRING,             // Section heading (for level 1+)
    token_count: INTEGER,
    char_start: INTEGER,         // Position in source article
    char_end: INTEGER,
    embedding: LIST<FLOAT>       // 1536-dim vector (ada-002)
})

// ═══════════════════════════════════════════════════════════════════════════
// ENTITY NODES (Dual-label pattern: :Entity:TypeLabel)
// ═══════════════════════════════════════════════════════════════════════════

// Base Entity properties (all types have these)
// {
//     id: STRING,              // Primary key, generated UUID
//     name: STRING,            // NORMALIZED: toLower(trim(extracted_name))
//     display_name: STRING,    // Original casing for UI display
//     confidence: FLOAT        // Extraction confidence 0.0-1.0
// }

(:Entity:Concept {
    id, name, display_name, confidence,
    definition: STRING           // What it means
})

(:Entity:Challenge {
    id, name, display_name, confidence,
    impact: STRING               // Business/technical impact
})

(:Entity:Artifact {
    id, name, display_name, confidence,
    definition: STRING
})

(:Entity:Bestpractice {
    id, name, display_name, confidence,
    benefit: STRING              // Why it helps
})

(:Entity:Processstage {
    id, name, display_name, confidence,
    definition: STRING
})

(:Entity:Role {
    id, name, display_name, confidence,
    organization: STRING         // Typical org context
})

(:Entity:Standard {
    id, name, display_name, confidence,
    organization: STRING         // ISO, IEC, FDA, etc.
})

(:Entity:Tool {
    id, name, display_name, confidence,
    vendor: STRING               // Company that makes it
})

(:Entity:Methodology {
    id, name, display_name, confidence,
    definition: STRING
})

(:Entity:Industry {
    id, name, display_name, confidence,
    definition: STRING
})

// ═══════════════════════════════════════════════════════════════════════════
// RESOURCE NODES (Flexible pattern: :Resource:TypeLabel)
// ═══════════════════════════════════════════════════════════════════════════

// Base Resource properties
// {
//     id: STRING,              // SHA256(normalized_url)[:16] or generated
//     url: STRING,             // Primary deduplication key
//     resource_type: STRING    // For filtering without label queries
// }

(:Resource:Image {
    id, url, resource_type: "image",
    alt_text: STRING,
    caption: STRING
})

(:Resource:Webinar {
    id, url, resource_type: "webinar",
    title: STRING
})

(:Resource:Video {
    id, url, resource_type: "video",
    platform: STRING,            // "youtube", "vimeo"
    video_id: STRING,
    title: STRING
})

(:Resource:ExternalLink {
    id, url, resource_type: "external_link",
    domain: STRING,              // Extracted for filtering
    anchor_text: STRING          // How it was linked in content
})

(:Resource:Definition {
    id, resource_type: "definition",
    term: STRING,                // The term being defined
    definition_text: STRING,     // The definition content
    url: STRING                  // NULL if not linked to glossary
})
```

### 3.2 Relationship Types

```cypher
// ═══════════════════════════════════════════════════════════════════════════
// CONTENT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

(:Chapter)-[:CONTAINS]->(:Article)
(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:CHILD_OF]->(:Chunk)           // Hierarchical: L2→L1→L0
(:Article)-[:RELATED_TO]->(:Article)     // "Related Articles" sidebar links

// ═══════════════════════════════════════════════════════════════════════════
// ENTITY PROVENANCE (replaces source_article_id LIST property)
// ═══════════════════════════════════════════════════════════════════════════

(:Article)-[:MENTIONS {
    confidence: FLOAT
}]->(:Entity)

(:Chunk)-[:MENTIONS {
    confidence: FLOAT,
    char_start: INTEGER,         // Position in chunk text
    char_end: INTEGER
}]->(:Entity)

// ═══════════════════════════════════════════════════════════════════════════
// GLOSSARY LINKING
// ═══════════════════════════════════════════════════════════════════════════

(:Article)-[:REFERENCES_TERM]->(:GlossaryTerm)
(:Chunk)-[:MENTIONS_TERM {
    char_start: INTEGER,
    char_end: INTEGER
}]->(:GlossaryTerm)
(:Entity)-[:DEFINED_BY]->(:GlossaryTerm)
(:Entity)-[:RELATED_TO_TERM]->(:GlossaryTerm)

// ═══════════════════════════════════════════════════════════════════════════
// ENTITY-TO-ENTITY (from neo4j_graphrag PATTERNS)
// ═══════════════════════════════════════════════════════════════════════════

// All entity relationships have these properties:
// {
//     confidence: FLOAT,       // Extraction confidence
//     evidence: STRING         // Source text supporting the relationship
// }

(:Entity)-[:ADDRESSES]->(:Entity)        // Solution addresses challenge
(:Entity)-[:REQUIRES]->(:Entity)         // Dependency relationship
(:Entity)-[:COMPONENT_OF]->(:Entity)     // Part-whole relationship
(:Entity)-[:RELATED_TO]->(:Entity)       // General semantic relationship
(:Entity)-[:ALTERNATIVE_TO]->(:Entity)   // Competing approaches
(:Entity)-[:USED_BY]->(:Entity)          // Tool/practice used by role/industry
(:Entity)-[:APPLIES_TO]->(:Entity)       // Standard applies to industry
(:Entity)-[:PRODUCES]->(:Entity)         // Process produces artifact
(:Entity)-[:DEFINES]->(:Entity)          // Defines another concept
(:Entity)-[:PREREQUISITE_FOR]->(:Entity) // Must come before

// ═══════════════════════════════════════════════════════════════════════════
// RESOURCE RELATIONSHIPS
// ═══════════════════════════════════════════════════════════════════════════

(:Article)-[:HAS_IMAGE {
    position: INTEGER,           // Order of appearance
    context: STRING              // "hero" | "inline" | "figure"
}]->(:Resource:Image)

(:Article)-[:PROMOTES_WEBINAR]->(:Resource:Webinar)

(:Article)-[:EMBEDS_VIDEO {
    position: INTEGER
}]->(:Resource:Video)

(:Article)-[:HAS_DEFINITION {
    position: INTEGER
}]->(:Resource:Definition)

(:Chunk)-[:CONTAINS_LINK {
    char_start: INTEGER,
    char_end: INTEGER,
    anchor_text: STRING
}]->(:Resource:ExternalLink)

// Link internal references (resolved during load)
(:Chunk)-[:LINKS_TO {
    char_start: INTEGER,
    char_end: INTEGER,
    anchor_text: STRING
}]->(:Article)

(:Chunk)-[:LINKS_TO {
    char_start: INTEGER,
    char_end: INTEGER,
    anchor_text: STRING
}]->(:GlossaryTerm)

// Definition → Glossary linking
(:Resource:Definition)-[:DEFINES_TERM]->(:GlossaryTerm)
```

### 3.3 Indexes and Constraints

```cypher
// ═══════════════════════════════════════════════════════════════════════════
// UNIQUENESS CONSTRAINTS (Primary Keys)
// ═══════════════════════════════════════════════════════════════════════════

CREATE CONSTRAINT chapter_pk IF NOT EXISTS
FOR (c:Chapter) REQUIRE c.chapter_number IS UNIQUE;

CREATE CONSTRAINT article_pk IF NOT EXISTS
FOR (a:Article) REQUIRE a.article_id IS UNIQUE;

CREATE CONSTRAINT chunk_pk IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_pk IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT glossary_pk IF NOT EXISTS
FOR (g:GlossaryTerm) REQUIRE g.term IS UNIQUE;

CREATE CONSTRAINT resource_pk IF NOT EXISTS
FOR (r:Resource) REQUIRE r.id IS UNIQUE;

// ═══════════════════════════════════════════════════════════════════════════
// PERFORMANCE INDEXES
// ═══════════════════════════════════════════════════════════════════════════

// Entity lookups by normalized name (critical for deduplication)
CREATE INDEX entity_name_idx IF NOT EXISTS
FOR (e:Entity) ON (e.name);

// Entity filtering by type
CREATE INDEX entity_type_idx IF NOT EXISTS
FOR (e:Entity) ON (e.display_name);

// Chunk hierarchical traversal
CREATE INDEX chunk_level_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.level);

// Resource URL deduplication
CREATE INDEX resource_url_idx IF NOT EXISTS
FOR (r:Resource) ON (r.url);

// Resource type filtering
CREATE INDEX resource_type_idx IF NOT EXISTS
FOR (r:Resource) ON (r.resource_type);

// ═══════════════════════════════════════════════════════════════════════════
// FULLTEXT INDEXES (for search)
// ═══════════════════════════════════════════════════════════════════════════

CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text, c.heading];

CREATE FULLTEXT INDEX article_content_fulltext IF NOT EXISTS
FOR (a:Article) ON EACH [a.title, a.summary];

CREATE FULLTEXT INDEX entity_search_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.display_name, e.definition];

CREATE FULLTEXT INDEX glossary_search_fulltext IF NOT EXISTS
FOR (g:GlossaryTerm) ON EACH [g.display_term, g.definition];

// ═══════════════════════════════════════════════════════════════════════════
// VECTOR INDEX (for semantic search)
// ═══════════════════════════════════════════════════════════════════════════

CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};
```

---

## 4. Data Quality Requirements

### 4.1 Entity Name Normalization

**Every entity name MUST be normalized before MERGE operations.**

```python
import re
import unicodedata

def normalize_entity_name(raw_name: str) -> str:
    """Normalize entity name for deduplication.
    
    Transformations:
    1. Strip leading/trailing whitespace
    2. Normalize unicode (NFKC)
    3. Convert to lowercase
    4. Collapse multiple spaces
    5. Preserve meaningful punctuation (hyphens, ampersands)
    
    Args:
        raw_name: Raw extracted entity name
        
    Returns:
        Normalized name for use as MERGE key
        
    Examples:
        "Artificial Intelligence (AI)" → "artificial intelligence (ai)"
        "  Traceability  " → "traceability"
        "V&V" → "v&v"
        "Benefit-Risk Analysis" → "benefit-risk analysis"
    """
    if not raw_name:
        return ""
    
    # Normalize unicode
    normalized = unicodedata.normalize("NFKC", raw_name)
    
    # Strip and lowercase
    normalized = normalized.strip().lower()
    
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized
```

**Cypher pattern for MERGE:**

```cypher
// CORRECT: Normalize during MERGE, preserve display_name
MERGE (e:Entity {name: toLower(trim($raw_name))})
ON CREATE SET 
    e.id = $entity_id,
    e.display_name = $raw_name,
    e.confidence = $confidence
ON MATCH SET
    e.display_name = CASE 
        WHEN size($raw_name) > size(e.display_name) THEN $raw_name 
        ELSE e.display_name 
    END
```

### 4.2 No LIST Properties for Identity Fields

**CRITICAL:** Never allow identity or core metadata properties to become LISTs.

```python
# BAD: This creates list properties
MERGE (e:Entity {name: $name})
ON MATCH SET e += $properties  # If $properties has different values, creates lists

# GOOD: Explicit property handling
MERGE (e:Entity {name: $normalized_name})
ON CREATE SET
    e.id = $id,
    e.display_name = $display_name,
    e.definition = $definition,
    e.confidence = $confidence
ON MATCH SET
    // Only update if new value is "better"
    e.definition = COALESCE(e.definition, $definition),
    e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END
```

### 4.3 Strict Property Types

| Property | Type | Enforcement |
|----------|------|-------------|
| `id` | STRING | Constraint + validation |
| `name` | STRING | Validation on write |
| `display_name` | STRING | Validation on write |
| `confidence` | FLOAT | Cast on write |
| `definition` | STRING | Validation on write |
| All relationship properties | SCALAR only | Validation on write |

### 4.4 Validation Query for LIST Properties

Run after every load to ensure no accidental LISTs:

```cypher
// Detect any LIST-type name properties
MATCH (n)
WHERE n.name IS NOT NULL AND n.name IS :: LIST<ANY>
RETURN labels(n) AS labels, n.name AS problematic_value, count(*) AS count;

// Should return 0 rows
```

---

## 5. Module Specifications

### 5.1 Project Structure

```
jama-guide-scraper/
├── pyproject.toml
├── README.md
├── CLAUDE.md                        # Update with new architecture
├── .env.example
├── src/
│   └── jama_scraper/
│       ├── __init__.py
│       │
│       │   # ═══ PHASE 1: SCRAPING (Keep + Extend) ═══
│       ├── scraper.py               # KEEP: Async orchestration
│       ├── fetcher.py               # KEEP: HTTP fetching
│       ├── parser.py                # EXTEND: Add resource extraction
│       ├── config.py                # KEEP: URL configs
│       │
│       │   # ═══ MODELS ═══
│       ├── models/
│       │   ├── __init__.py
│       │   ├── content.py           # Article, Chapter, Section
│       │   ├── chunk.py             # Chunk models (hierarchical)
│       │   ├── entity.py            # Entity models
│       │   ├── resource.py          # NEW: Resource models
│       │   └── glossary.py          # GlossaryTerm
│       │
│       │   # ═══ PHASE 2: CHUNKING (Refactor) ═══
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── hierarchical_chunker.py  # NEW: 3-tier chunking
│       │   ├── config.py                # Chunking configuration
│       │   └── text_utils.py            # Token counting, splitting
│       │
│       │   # ═══ PHASE 3: EXTRACTION (Replace) ═══
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── schema.py            # NEW: neo4j_graphrag schema definition
│       │   ├── pipeline.py          # NEW: SimpleKGPipeline wrapper
│       │   └── prompts.py           # LLM prompts for extraction
│       │
│       │   # ═══ PHASE 4: POST-PROCESSING (New) ═══
│       ├── postprocessing/
│       │   ├── __init__.py
│       │   ├── industry_taxonomy.py     # Industry mapping (96→19)
│       │   ├── entity_validator.py      # Type validation/reclassification
│       │   ├── glossary_linker.py       # Auto-link entities to glossary
│       │   └── normalizer.py            # Name normalization utilities
│       │
│       │   # ═══ PHASE 5: EMBEDDING (Keep) ═══
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── embedder.py          # KEEP: OpenAI embeddings
│       │   └── config.py            # Embedding configuration
│       │
│       │   # ═══ PHASE 6: GRAPH WRITING (Replace) ═══
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── writer.py            # NEW: neo4j_graphrag writer integration
│       │   ├── supplementary.py     # NEW: Supplementary Cypher queries
│       │   └── schema.py            # NEW: Constraint/index creation
│       │
│       │   # ═══ PHASE 7: VALIDATION (New) ═══
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── queries.py           # Validation Cypher queries
│       │   └── reporter.py          # Validation result reporting
│       │
│       │   # ═══ CLI ═══
│       ├── cli.py                   # Updated CLI
│       └── exceptions.py            # Custom exceptions
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── test_normalization.py
│   ├── test_industry_taxonomy.py
│   ├── test_chunking.py
│   ├── test_extraction.py
│   └── test_integration.py
│
└── output/                          # Generated files
    ├── scraped/                     # Raw scraped data
    ├── processed/                   # Processed intermediate data
    └── validation/                  # Validation reports
```

### 5.2 Module: `models/resource.py`

```python
"""Resource models for flexible content type handling.

Implements the extensible Resource pattern with type-specific models
for images, webinars, videos, links, and definitions.
"""

from __future__ import annotations

import hashlib
from abc import ABC
from enum import Enum
from typing import ClassVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field


class ResourceType(str, Enum):
    """Enumeration of supported resource types."""
    IMAGE = "image"
    WEBINAR = "webinar"
    VIDEO = "video"
    EXTERNAL_LINK = "external_link"
    DEFINITION = "definition"


class BaseResource(BaseModel, ABC):
    """Abstract base class for all resource types.
    
    All resources have:
    - A unique ID (derived from URL or generated)
    - A resource_type discriminator
    - Optional URL (some resources like definitions may not have URLs)
    """
    
    resource_type: ClassVar[ResourceType]
    url: str | None = None
    
    @computed_field
    @property
    def id(self) -> str:
        """Generate stable ID from URL or content hash."""
        if self.url:
            return self._hash_url(self.url)
        return self._generate_content_hash()
    
    @staticmethod
    def _hash_url(url: str) -> str:
        """Generate stable ID from normalized URL."""
        # Normalize: remove trailing slash, lowercase
        normalized = url.rstrip("/").lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _generate_content_hash(self) -> str:
        """Generate ID from content (for URL-less resources)."""
        content = self.model_dump_json(exclude={"id"})
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @computed_field
    @property
    def neo4j_labels(self) -> list[str]:
        """Neo4j labels for this resource type."""
        type_label = self.resource_type.value.title().replace("_", "")
        return ["Resource", type_label]


class ImageResource(BaseResource):
    """Content image (excludes decorative images)."""
    
    resource_type: ClassVar[ResourceType] = ResourceType.IMAGE
    url: str
    alt_text: str | None = None
    caption: str | None = None
    
    @computed_field
    @property
    def is_decorative(self) -> bool:
        """Check if image appears to be decorative.
        
        Decorative images are filtered out during extraction.
        """
        decorative_patterns = [
            "logo", "icon", "arrow", "bullet", "spacer",
            "background", "decoration", "ornament"
        ]
        
        if self.alt_text:
            alt_lower = self.alt_text.lower()
            return any(p in alt_lower for p in decorative_patterns)
        
        # No alt text often indicates decorative
        url_lower = self.url.lower()
        return any(p in url_lower for p in decorative_patterns)


class WebinarResource(BaseResource):
    """Webinar promotion link."""
    
    resource_type: ClassVar[ResourceType] = ResourceType.WEBINAR
    url: str
    title: str | None = None


class VideoResource(BaseResource):
    """Embedded video (YouTube, Vimeo, etc.)."""
    
    resource_type: ClassVar[ResourceType] = ResourceType.VIDEO
    url: str
    platform: str  # "youtube", "vimeo", "wistia"
    video_id: str
    title: str | None = None
    
    @classmethod
    def from_youtube_url(cls, url: str, title: str | None = None) -> VideoResource:
        """Create from YouTube URL."""
        # Extract video ID from various YouTube URL formats
        parsed = urlparse(url)
        
        if "youtu.be" in parsed.netloc:
            video_id = parsed.path.lstrip("/")
        elif "youtube.com" in parsed.netloc:
            from urllib.parse import parse_qs
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            video_id = ""
        
        return cls(
            url=url,
            platform="youtube",
            video_id=video_id,
            title=title
        )


class ExternalLinkResource(BaseResource):
    """External link found in content."""
    
    resource_type: ClassVar[ResourceType] = ResourceType.EXTERNAL_LINK
    url: str
    anchor_text: str
    
    @computed_field
    @property
    def domain(self) -> str:
        """Extract domain for filtering/grouping."""
        parsed = urlparse(self.url)
        return parsed.netloc.lower()


class DefinitionResource(BaseResource):
    """Definition block (Avia message box)."""
    
    resource_type: ClassVar[ResourceType] = ResourceType.DEFINITION
    term: str
    definition_text: str
    url: str | None = None  # Link to glossary if present
    
    @computed_field
    @property
    def normalized_term(self) -> str:
        """Normalized term for glossary matching."""
        return self.term.lower().strip()


# Type alias for any resource
Resource = ImageResource | WebinarResource | VideoResource | ExternalLinkResource | DefinitionResource


# Registry for extensibility
RESOURCE_TYPE_REGISTRY: dict[ResourceType, type[BaseResource]] = {
    ResourceType.IMAGE: ImageResource,
    ResourceType.WEBINAR: WebinarResource,
    ResourceType.VIDEO: VideoResource,
    ResourceType.EXTERNAL_LINK: ExternalLinkResource,
    ResourceType.DEFINITION: DefinitionResource,
}
```

### 5.3 Module: `parser.py` Extensions

Add resource extraction to the existing parser:

```python
"""HTML parser extensions for resource extraction.

Add to existing parser.py:
- extract_images() - Content images only
- extract_webinars() - Webinar promotion links
- extract_videos() - Embedded videos
- extract_definitions() - Definition message boxes
- extract_external_links() - Links in content
- extract_related_articles() - Related article links
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from .models.resource import (
    DefinitionResource,
    ExternalLinkResource,
    ImageResource,
    VideoResource,
    WebinarResource,
)

if TYPE_CHECKING:
    from .models.content import RelatedArticleLink


class ResourceExtractor:
    """Extract resources from article HTML."""
    
    # Decorative image patterns (filter out)
    DECORATIVE_PATTERNS = re.compile(
        r'(logo|icon|arrow|bullet|spacer|background|decoration|'
        r'ornament|avatar|placeholder|loading|spinner)',
        re.IGNORECASE
    )
    
    # Jama domain for internal link detection
    JAMA_DOMAIN = "jamasoftware.com"
    
    def __init__(self, base_url: str):
        """Initialize with article base URL for relative link resolution."""
        self.base_url = base_url
    
    def extract_all(self, soup: BeautifulSoup) -> dict[str, list]:
        """Extract all resource types from article HTML.
        
        Returns:
            Dictionary with keys: images, webinars, videos, definitions,
            external_links, related_articles
        """
        # Find the main content area (exclude header, footer, sidebar)
        content = soup.select_one("article, .entry-content, main")
        if not content:
            content = soup
        
        return {
            "images": self.extract_images(content),
            "webinars": self.extract_webinars(soup),  # May be in sidebar
            "videos": self.extract_videos(content),
            "definitions": self.extract_definitions(content),
            "external_links": self.extract_external_links(content),
            "related_articles": self.extract_related_articles(soup),
        }
    
    def extract_images(self, content: Tag) -> list[ImageResource]:
        """Extract content images, filtering out decorative ones."""
        images = []
        
        for img in content.select("img"):
            src = img.get("src", "")
            if not src:
                continue
            
            # Resolve relative URLs
            url = urljoin(self.base_url, src)
            
            alt_text = img.get("alt", "")
            
            # Check for decorative patterns
            if self._is_decorative_image(url, alt_text, img):
                continue
            
            # Extract caption from figure/figcaption
            caption = None
            figure = img.find_parent("figure")
            if figure:
                figcaption = figure.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text(strip=True)
            
            images.append(ImageResource(
                url=url,
                alt_text=alt_text or None,
                caption=caption
            ))
        
        return images
    
    def _is_decorative_image(self, url: str, alt_text: str, img: Tag) -> bool:
        """Determine if an image is decorative (should be filtered)."""
        # Check URL patterns
        if self.DECORATIVE_PATTERNS.search(url):
            return True
        
        # Check alt text patterns
        if alt_text and self.DECORATIVE_PATTERNS.search(alt_text):
            return True
        
        # Check CSS classes
        classes = " ".join(img.get("class", []))
        if self.DECORATIVE_PATTERNS.search(classes):
            return True
        
        # Very small images are likely decorative
        width = img.get("width", "")
        height = img.get("height", "")
        try:
            if width and int(width) < 50:
                return True
            if height and int(height) < 50:
                return True
        except ValueError:
            pass
        
        # Empty alt with role="presentation" is explicitly decorative
        if img.get("role") == "presentation":
            return True
        
        return False
    
    def extract_webinars(self, soup: BeautifulSoup) -> list[WebinarResource]:
        """Extract webinar promotion links."""
        webinars = []
        
        # Common webinar CTA patterns
        selectors = [
            ".webinar-cta a",
            "[data-webinar] a",
            'a[href*="webinar"]',
            'a[href*="on-demand"]',
            ".cta-webinar a",
        ]
        
        seen_urls = set()
        for selector in selectors:
            for link in soup.select(selector):
                href = link.get("href", "")
                if not href or href in seen_urls:
                    continue
                
                url = urljoin(self.base_url, href)
                seen_urls.add(url)
                
                # Extract title from link text or nearby heading
                title = link.get_text(strip=True)
                if not title:
                    heading = link.find_previous(["h2", "h3", "h4"])
                    if heading:
                        title = heading.get_text(strip=True)
                
                webinars.append(WebinarResource(
                    url=url,
                    title=title or None
                ))
        
        return webinars
    
    def extract_videos(self, content: Tag) -> list[VideoResource]:
        """Extract embedded videos."""
        videos = []
        
        # YouTube iframes
        for iframe in content.select('iframe[src*="youtube"], iframe[src*="youtu.be"]'):
            src = iframe.get("src", "")
            title = iframe.get("title")
            videos.append(VideoResource.from_youtube_url(src, title))
        
        # Vimeo iframes
        for iframe in content.select('iframe[src*="vimeo"]'):
            src = iframe.get("src", "")
            title = iframe.get("title")
            
            # Extract Vimeo ID
            match = re.search(r'vimeo\.com/(?:video/)?(\d+)', src)
            video_id = match.group(1) if match else ""
            
            videos.append(VideoResource(
                url=src,
                platform="vimeo",
                video_id=video_id,
                title=title
            ))
        
        return videos
    
    def extract_definitions(self, content: Tag) -> list[DefinitionResource]:
        """Extract definition blocks (Avia message boxes).
        
        Target HTML pattern:
        <div class="avia_message_box ...">
            <span class="avia_message_box_title">DEFINITION OF TERM:</span>
            <div class="avia_message_box_content">
                <p><strong>Term</strong> is definition text...</p>
            </div>
        </div>
        """
        definitions = []
        
        for box in content.select(".avia_message_box"):
            # Extract term from title
            title_elem = box.select_one(".avia_message_box_title")
            if not title_elem:
                continue
            
            title_text = title_elem.get_text(strip=True)
            
            # Parse "DEFINITION OF X:" pattern
            match = re.match(r"DEFINITION\s+OF\s+(.+?):", title_text, re.IGNORECASE)
            if not match:
                continue
            
            term = match.group(1).strip()
            
            # Extract definition content
            content_elem = box.select_one(".avia_message_box_content")
            if not content_elem:
                continue
            
            definition_text = content_elem.get_text(strip=True)
            
            # Check for link to glossary
            glossary_link = content_elem.select_one('a[href*="glossary"]')
            url = None
            if glossary_link:
                url = urljoin(self.base_url, glossary_link.get("href", ""))
            
            definitions.append(DefinitionResource(
                term=term,
                definition_text=definition_text,
                url=url
            ))
        
        return definitions
    
    def extract_external_links(self, content: Tag) -> list[ExternalLinkResource]:
        """Extract external links from content."""
        links = []
        seen_urls = set()
        
        for a in content.select("a[href]"):
            href = a.get("href", "")
            if not href or href.startswith("#"):
                continue
            
            url = urljoin(self.base_url, href)
            parsed = urlparse(url)
            
            # Skip internal links
            if self.JAMA_DOMAIN in parsed.netloc:
                continue
            
            # Skip mailto/tel links
            if parsed.scheme in ("mailto", "tel"):
                continue
            
            # Deduplicate
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            anchor_text = a.get_text(strip=True)
            
            links.append(ExternalLinkResource(
                url=url,
                anchor_text=anchor_text
            ))
        
        return links
    
    def extract_related_articles(
        self, 
        soup: BeautifulSoup,
        article_index: dict[str, str] | None = None
    ) -> list[tuple[str, str | None]]:
        """Extract related article links.
        
        Args:
            soup: Full page HTML
            article_index: Optional mapping of URL paths to article_ids
        
        Returns:
            List of (url, resolved_article_id) tuples
        """
        related = []
        
        # Common "related articles" section patterns
        selectors = [
            ".related-articles a",
            ".related-posts a",
            '[class*="related"] a',
            ".sidebar a[href*='requirements-management-guide']",
        ]
        
        seen_urls = set()
        for selector in selectors:
            for link in soup.select(selector):
                href = link.get("href", "")
                if not href or href in seen_urls:
                    continue
                
                url = urljoin(self.base_url, href)
                
                # Only include links to the requirements guide
                if "requirements-management-guide" not in url:
                    continue
                
                seen_urls.add(url)
                
                # Resolve to article_id if index provided
                article_id = None
                if article_index:
                    path = urlparse(url).path.rstrip("/")
                    article_id = article_index.get(path)
                
                related.append((url, article_id))
        
        return related
```

---

## 6. neo4j_graphrag Integration

### 6.1 Schema Definition

```python
"""neo4j_graphrag schema definition for requirements management domain.

Defines NODE_TYPES, RELATIONSHIP_TYPES, and PATTERNS that constrain
LLM extraction to produce a consistent knowledge graph.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# NODE TYPES
# =============================================================================
# Each node type includes a label, description (for LLM guidance), and properties.
# The LLM uses descriptions to classify extracted entities.

NODE_TYPES: list[dict[str, Any]] = [
    {
        "label": "Concept",
        "description": (
            "Technical concepts, terminology, and abstract ideas in requirements "
            "management. Examples: requirements traceability, verification, "
            "baseline management, EARS notation, bidirectional traceability."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
    {
        "label": "Challenge",
        "description": (
            "Problems, difficulties, obstacles, and risks that teams face in "
            "requirements management. Examples: scope creep, requirement ambiguity, "
            "poor traceability, compliance gaps, change impact uncertainty."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "impact", "type": "STRING"},
        ],
    },
    {
        "label": "Artifact",
        "description": (
            "Documents, work products, and deliverables created during product "
            "development. Examples: SRS, PRD, traceability matrix, DHF, "
            "design specification, test plan, verification report."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
    {
        "label": "Bestpractice",
        "description": (
            "Recommended practices, solutions, and approaches for effective "
            "requirements management. Examples: live traceability, baseline "
            "management, requirements reviews, impact analysis, version control."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "benefit", "type": "STRING"},
        ],
    },
    {
        "label": "Processstage",
        "description": (
            "Stages, phases, and activities in the product development lifecycle. "
            "Examples: requirements elicitation, verification, validation, "
            "change control, design review, acceptance testing."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
    {
        "label": "Role",
        "description": (
            "Job roles, stakeholders, and participants in requirements management. "
            "Examples: product manager, systems engineer, QA engineer, "
            "regulatory affairs, business analyst, project manager."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "organization", "type": "STRING"},
        ],
    },
    {
        "label": "Standard",
        "description": (
            "Industry standards, regulations, and compliance frameworks. "
            "Examples: ISO 13485, DO-178C, IEC 62304, FDA 21 CFR Part 820, "
            "ISO 26262, ASPICE, MIL-STD-498."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "organization", "type": "STRING"},
        ],
    },
    {
        "label": "Tool",
        "description": (
            "Software tools, platforms, and systems used for requirements "
            "management. Examples: Jama Connect, DOORS, Polarion, Jira, "
            "Confluence, Excel, ReqIF."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "vendor", "type": "STRING"},
        ],
    },
    {
        "label": "Methodology",
        "description": (
            "Development methodologies, frameworks, and approaches. "
            "Examples: Agile, V-Model, MBSE, Waterfall, SAFe, Scrum, "
            "DevOps, Continuous Integration."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
    {
        "label": "Industry",
        "description": (
            "Industry verticals and market sectors. IMPORTANT: Only extract "
            "actual industries, NOT technology concepts. Valid examples: "
            "aerospace, automotive, medical devices, defense, semiconductor. "
            "INVALID (these are Concepts, not Industries): AI, machine learning, "
            "cloud computing, IoT, software development."
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "definition", "type": "STRING"},
        ],
    },
]


# =============================================================================
# RELATIONSHIP TYPES
# =============================================================================
# Define semantic edge labels with descriptions for LLM guidance.

RELATIONSHIP_TYPES: list[dict[str, Any]] = [
    {
        "label": "ADDRESSES",
        "description": (
            "A practice, tool, or concept that addresses or solves a challenge. "
            "Example: Live traceability ADDRESSES the challenge of outdated links."
        ),
    },
    {
        "label": "REQUIRES",
        "description": (
            "A standard, process, or concept that requires another element. "
            "Example: ISO 13485 REQUIRES documented design controls."
        ),
    },
    {
        "label": "COMPONENT_OF",
        "description": (
            "A part-whole or hierarchical membership relationship. "
            "Example: Verification is a COMPONENT_OF V&V."
        ),
    },
    {
        "label": "RELATED_TO",
        "description": (
            "A general semantic relationship between concepts. "
            "Example: Traceability is RELATED_TO change management."
        ),
    },
    {
        "label": "ALTERNATIVE_TO",
        "description": (
            "Competing or alternative approaches. "
            "Example: Live traceability is an ALTERNATIVE_TO after-the-fact traceability."
        ),
    },
    {
        "label": "USED_BY",
        "description": (
            "A tool, practice, or methodology used by a role or industry. "
            "Example: MBSE is USED_BY aerospace companies."
        ),
    },
    {
        "label": "APPLIES_TO",
        "description": (
            "A standard or regulation that applies to an industry or artifact. "
            "Example: DO-178C APPLIES_TO aerospace software."
        ),
    },
    {
        "label": "PRODUCES",
        "description": (
            "A process or activity that produces an artifact. "
            "Example: Requirements elicitation PRODUCES the SRS."
        ),
    },
    {
        "label": "DEFINES",
        "description": (
            "A standard or concept that defines another element. "
            "Example: V-Model DEFINES verification stages."
        ),
    },
    {
        "label": "PREREQUISITE_FOR",
        "description": (
            "A concept or activity that must precede another. "
            "Example: Requirements definition is a PREREQUISITE_FOR verification."
        ),
    },
]


# =============================================================================
# PATTERNS (Valid relationship triples)
# =============================================================================
# Constrains which (source_label, relationship, target_label) combinations
# the LLM is allowed to create. Prevents semantically invalid relationships.

PATTERNS: list[tuple[str, str, str]] = [
    # Concept relationships
    ("Concept", "RELATED_TO", "Concept"),
    ("Concept", "COMPONENT_OF", "Concept"),
    ("Concept", "PREREQUISITE_FOR", "Concept"),
    ("Concept", "ADDRESSES", "Challenge"),
    ("Concept", "DEFINES", "Artifact"),
    
    # Challenge relationships
    ("Challenge", "RELATED_TO", "Challenge"),
    ("Challenge", "COMPONENT_OF", "Challenge"),
    
    # Bestpractice relationships
    ("Bestpractice", "ADDRESSES", "Challenge"),
    ("Bestpractice", "RELATED_TO", "Concept"),
    ("Bestpractice", "PRODUCES", "Artifact"),
    ("Bestpractice", "COMPONENT_OF", "Methodology"),
    
    # Processstage relationships
    ("Processstage", "PRODUCES", "Artifact"),
    ("Processstage", "PREREQUISITE_FOR", "Processstage"),
    ("Processstage", "COMPONENT_OF", "Methodology"),
    ("Processstage", "ADDRESSES", "Challenge"),
    
    # Tool relationships
    ("Tool", "ADDRESSES", "Challenge"),
    ("Tool", "USED_BY", "Role"),
    ("Tool", "USED_BY", "Industry"),
    ("Tool", "RELATED_TO", "Concept"),
    
    # Methodology relationships
    ("Methodology", "ADDRESSES", "Challenge"),
    ("Methodology", "DEFINES", "Processstage"),
    ("Methodology", "USED_BY", "Industry"),
    ("Methodology", "ALTERNATIVE_TO", "Methodology"),
    
    # Standard relationships
    ("Standard", "APPLIES_TO", "Industry"),
    ("Standard", "REQUIRES", "Concept"),
    ("Standard", "REQUIRES", "Artifact"),
    ("Standard", "REQUIRES", "Processstage"),
    ("Standard", "DEFINES", "Processstage"),
    ("Standard", "RELATED_TO", "Standard"),
    
    # Role relationships
    ("Role", "PRODUCES", "Artifact"),
    ("Role", "COMPONENT_OF", "Processstage"),
    
    # Industry relationships
    ("Industry", "RELATED_TO", "Industry"),
    
    # Artifact relationships
    ("Artifact", "COMPONENT_OF", "Artifact"),
    ("Artifact", "RELATED_TO", "Concept"),
]


def get_schema_dict() -> dict[str, Any]:
    """Return schema dictionary for SimpleKGPipeline."""
    return {
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
    }
```

### 6.2 Pipeline Wrapper

```python
"""neo4j_graphrag pipeline wrapper for Jama guide processing.

Configures SimpleKGPipeline with domain-specific schema and
custom text splitter for HTML content.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from .schema import get_schema_dict

if TYPE_CHECKING:
    from neo4j import Driver

load_dotenv()


class JamaKGPipeline:
    """Knowledge graph pipeline for Jama requirements guide.
    
    Wraps neo4j_graphrag's SimpleKGPipeline with domain-specific
    configuration for requirements management content.
    """
    
    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        neo4j_database: str | None = None,
        openai_api_key: str | None = None,
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """Initialize the pipeline.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            openai_api_key: OpenAI API key
            llm_model: LLM model for extraction
            embedding_model: Embedding model for vectors
        """
        # Load from environment if not provided
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self._validate_config()
        
        # Initialize components
        self._driver: Driver | None = None
        self._llm: OpenAILLM | None = None
        self._embedder: OpenAIEmbeddings | None = None
        self._pipeline: SimpleKGPipeline | None = None
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required = [
            ("NEO4J_URI", self.neo4j_uri),
            ("NEO4J_USERNAME", self.neo4j_username),
            ("NEO4J_PASSWORD", self.neo4j_password),
            ("NEO4J_DATABASE", self.neo4j_database),
            ("OPENAI_API_KEY", self.openai_api_key),
        ]
        
        missing = [name for name, value in required if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
    
    @property
    def driver(self) -> Driver:
        """Get or create Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password),
            )
            self._driver.verify_connectivity()
        return self._driver
    
    @property
    def llm(self) -> OpenAILLM:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = OpenAILLM(
                model_name=self.llm_model,
                model_params={
                    "temperature": 0,  # Deterministic for consistent extraction
                    "response_format": {"type": "json_object"},
                },
            )
        return self._llm
    
    @property
    def embedder(self) -> OpenAIEmbeddings:
        """Get or create embedder instance."""
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(model=self.embedding_model)
        return self._embedder
    
    def get_pipeline(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> SimpleKGPipeline:
        """Get configured KG pipeline.
        
        Args:
            chunk_size: Characters per chunk for text splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            Configured SimpleKGPipeline
        """
        text_splitter = FixedSizeSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        return SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            neo4j_database=self.neo4j_database,
            embedder=self.embedder,
            from_pdf=False,  # We're processing HTML/markdown
            text_splitter=text_splitter,
            schema=get_schema_dict(),
        )
    
    async def process_article(
        self,
        article_id: str,
        content: str,
    ) -> dict:
        """Process a single article through the pipeline.
        
        Args:
            article_id: Unique article identifier
            content: Article content (markdown or text)
            
        Returns:
            Pipeline result with extraction statistics
        """
        pipeline = self.get_pipeline()
        
        # Note: SimpleKGPipeline expects file paths; we may need to
        # write content to temp file or use a custom component
        # This is a placeholder for the actual implementation
        
        result = await pipeline.run_async(text=content)
        return result
    
    def close(self) -> None:
        """Close connections."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

### 6.3 Supplementary Cypher

```python
"""Supplementary Cypher queries for domain-specific graph structure.

After SimpleKGPipeline creates base nodes, run these queries to:
1. Create Chapter/Article content structure
2. Add :Entity base label to type-specific nodes
3. Link definitions to glossary terms
4. Create hierarchical chunk relationships
"""

from __future__ import annotations

# =============================================================================
# CHAPTER AND ARTICLE STRUCTURE
# =============================================================================

CREATE_CHAPTER_QUERY = """
MERGE (c:Chapter {chapter_number: $chapter_number})
SET c.title = $title,
    c.url = $url
"""

CREATE_ARTICLE_QUERY = """
MERGE (a:Article {article_id: $article_id})
SET a.title = $title,
    a.url = $url,
    a.summary = $summary,
    a.word_count = $word_count,
    a.chapter_number = $chapter_number
"""

LINK_CHAPTER_TO_ARTICLE = """
MATCH (c:Chapter {chapter_number: $chapter_number})
MATCH (a:Article {article_id: $article_id})
MERGE (c)-[:CONTAINS]->(a)
"""

# =============================================================================
# ENTITY BASE LABEL
# =============================================================================
# Add :Entity label to all type-specific entity nodes

ADD_ENTITY_LABEL_QUERY = """
MATCH (n)
WHERE n:Concept OR n:Challenge OR n:Artifact OR n:Bestpractice OR 
      n:Processstage OR n:Role OR n:Standard OR n:Tool OR 
      n:Methodology OR n:Industry
SET n:Entity
"""

# =============================================================================
# ENTITY NAME NORMALIZATION
# =============================================================================
# Ensure all entities have normalized name and display_name

NORMALIZE_ENTITY_NAMES = """
MATCH (e:Entity)
WHERE e.display_name IS NULL
SET e.display_name = e.name

WITH e
SET e.name = toLower(trim(e.name))
"""

# =============================================================================
# GLOSSARY TERM LINKING
# =============================================================================

CREATE_GLOSSARY_TERM = """
MERGE (g:GlossaryTerm {term: toLower(trim($term))})
SET g.display_term = $display_term,
    g.definition = $definition
"""

LINK_ENTITY_TO_GLOSSARY = """
MATCH (e:Entity)
WHERE toLower(trim(e.name)) = $normalized_term
MATCH (g:GlossaryTerm {term: $normalized_term})
MERGE (e)-[:DEFINED_BY]->(g)
"""

AUTO_LINK_DEFINITIONS_TO_GLOSSARY = """
MATCH (d:Resource:Definition)
MATCH (g:GlossaryTerm)
WHERE toLower(trim(d.term)) = g.term
MERGE (d)-[:DEFINES_TERM]->(g)
"""

# =============================================================================
# HIERARCHICAL CHUNK STRUCTURE
# =============================================================================

CREATE_CHUNK_QUERY = """
MERGE (c:Chunk {id: $id})
SET c.text = $text,
    c.level = $level,
    c.chunk_type = $chunk_type,
    c.heading = $heading,
    c.token_count = $token_count,
    c.char_start = $char_start,
    c.char_end = $char_end
"""

LINK_ARTICLE_TO_CHUNK = """
MATCH (a:Article {article_id: $article_id})
MATCH (c:Chunk {id: $chunk_id})
MERGE (a)-[:HAS_CHUNK]->(c)
"""

LINK_CHUNK_HIERARCHY = """
MATCH (child:Chunk {id: $child_id})
MATCH (parent:Chunk {id: $parent_id})
MERGE (child)-[:CHILD_OF]->(parent)
"""

SET_CHUNK_EMBEDDING = """
MATCH (c:Chunk {id: $chunk_id})
SET c.embedding = $embedding
"""

# =============================================================================
# RESOURCE CREATION
# =============================================================================

CREATE_IMAGE_RESOURCE = """
MERGE (r:Resource:Image {id: $id})
SET r.url = $url,
    r.resource_type = 'image',
    r.alt_text = $alt_text,
    r.caption = $caption
"""

CREATE_WEBINAR_RESOURCE = """
MERGE (r:Resource:Webinar {id: $id})
SET r.url = $url,
    r.resource_type = 'webinar',
    r.title = $title
"""

CREATE_VIDEO_RESOURCE = """
MERGE (r:Resource:Video {id: $id})
SET r.url = $url,
    r.resource_type = 'video',
    r.platform = $platform,
    r.video_id = $video_id,
    r.title = $title
"""

CREATE_EXTERNAL_LINK_RESOURCE = """
MERGE (r:Resource:ExternalLink {id: $id})
SET r.url = $url,
    r.resource_type = 'external_link',
    r.domain = $domain,
    r.anchor_text = $anchor_text
"""

CREATE_DEFINITION_RESOURCE = """
MERGE (r:Resource:Definition {id: $id})
SET r.resource_type = 'definition',
    r.term = $term,
    r.definition_text = $definition_text,
    r.url = $url
"""

# =============================================================================
# RESOURCE LINKING
# =============================================================================

LINK_ARTICLE_TO_IMAGE = """
MATCH (a:Article {article_id: $article_id})
MATCH (r:Resource:Image {id: $resource_id})
MERGE (a)-[:HAS_IMAGE {position: $position, context: $context}]->(r)
"""

LINK_ARTICLE_TO_WEBINAR = """
MATCH (a:Article {article_id: $article_id})
MATCH (r:Resource:Webinar {id: $resource_id})
MERGE (a)-[:PROMOTES_WEBINAR]->(r)
"""

LINK_ARTICLE_TO_VIDEO = """
MATCH (a:Article {article_id: $article_id})
MATCH (r:Resource:Video {id: $resource_id})
MERGE (a)-[:EMBEDS_VIDEO {position: $position}]->(r)
"""

LINK_ARTICLE_TO_DEFINITION = """
MATCH (a:Article {article_id: $article_id})
MATCH (r:Resource:Definition {id: $resource_id})
MERGE (a)-[:HAS_DEFINITION {position: $position}]->(r)
"""

LINK_CHUNK_TO_EXTERNAL_LINK = """
MATCH (c:Chunk {id: $chunk_id})
MATCH (r:Resource:ExternalLink {id: $resource_id})
MERGE (c)-[:CONTAINS_LINK {
    char_start: $char_start,
    char_end: $char_end,
    anchor_text: $anchor_text
}]->(r)
"""

# =============================================================================
# RELATED ARTICLES
# =============================================================================

LINK_RELATED_ARTICLES = """
MATCH (a1:Article {article_id: $source_article_id})
MATCH (a2:Article {article_id: $target_article_id})
WHERE a1 <> a2
MERGE (a1)-[:RELATED_TO]->(a2)
"""
```

### 6.4 Custom Extraction Prompts with Few-Shot Examples

The `entity_extraction_prompt.py` example demonstrates how to customize extraction using `ERExtractionTemplate`. This is **critical** for accurate Industry vs Concept classification.

```python
"""Custom extraction prompt with domain-specific few-shot examples.

Uses ERExtractionTemplate to prepend domain instructions that guide the LLM
to correctly classify entities in the requirements management domain.
"""

from neo4j_graphrag.generation.prompts import ERExtractionTemplate

# =============================================================================
# DOMAIN-SPECIFIC EXTRACTION INSTRUCTIONS
# =============================================================================
# These instructions are prepended to the default extraction template.
# They provide context and few-shot examples that dramatically improve
# extraction accuracy for our domain.
# =============================================================================

REQUIREMENTS_DOMAIN_INSTRUCTIONS = """
## DOMAIN CONTEXT: Requirements Management

You are extracting entities from content about requirements management, 
product development lifecycle, and regulatory compliance.

## CRITICAL CLASSIFICATION RULES

### Industry vs Concept Disambiguation
Technology concepts are NOT industries. Follow these rules strictly:

VALID INDUSTRIES (extract as "Industry"):
- Aerospace & Defense (includes: aerospace, defense, aviation, aircraft)
- Automotive (includes: automotive, vehicle, autonomous vehicles)
- Medical Devices (includes: medical device, medtech, healthcare devices)
- Pharmaceuticals & Life Sciences (includes: pharma, biotech, drug development)
- Semiconductor & Electronics (includes: semiconductor, chip design, electronics)
- Financial Services, Telecommunications, Energy & Utilities
- Industrial & Manufacturing, Rail & Transportation
- Government & Public Sector, Software & Technology

NOT INDUSTRIES - Extract as "Concept" instead:
- artificial intelligence, AI, machine learning, ML
- cloud computing, high-performance computing
- internet of things, IoT, cyber-physical systems
- automation, digital transformation
- software development, embedded systems

### Compound Industry Handling
- "aerospace and defense" → ONE Industry: "Aerospace & Defense"
- "medical device and life sciences" → ONE Industry: "Medical Devices"
- Do NOT split compound industries into separate entities

### Standard Organization Extraction
When extracting Standards, include the issuing organization:
- "ISO 26262" → Standard with organization: "ISO"
- "DO-178C" → Standard with organization: "RTCA"  
- "FDA 21 CFR Part 820" → Standard with organization: "FDA"

## FEW-SHOT EXAMPLES

INPUT: "DO-178C certification is required for aerospace software development."
OUTPUT: {
  "entities": [
    {"label": "Standard", "name": "DO-178C", "properties": {"organization": "RTCA"}},
    {"label": "Industry", "name": "Aerospace & Defense"}
  ],
  "relationships": [
    {"source": "DO-178C", "type": "APPLIES_TO", "target": "Aerospace & Defense"}
  ]
}

INPUT: "AI and machine learning are transforming the automotive industry."
OUTPUT: {
  "entities": [
    {"label": "Concept", "name": "artificial intelligence"},
    {"label": "Concept", "name": "machine learning"},
    {"label": "Industry", "name": "Automotive"}
  ],
  "relationships": [
    {"source": "artificial intelligence", "type": "USED_BY", "target": "Automotive"},
    {"source": "machine learning", "type": "USED_BY", "target": "Automotive"}
  ]
}

INPUT: "Requirements traceability addresses the challenge of change impact analysis."
OUTPUT: {
  "entities": [
    {"label": "Concept", "name": "requirements traceability"},
    {"label": "Challenge", "name": "change impact analysis"}
  ],
  "relationships": [
    {"source": "requirements traceability", "type": "ADDRESSES", "target": "change impact analysis"}
  ]
}

INPUT: "The V-model defines verification stages including unit testing and integration testing."
OUTPUT: {
  "entities": [
    {"label": "Methodology", "name": "V-model"},
    {"label": "Processstage", "name": "verification"},
    {"label": "Processstage", "name": "unit testing"},
    {"label": "Processstage", "name": "integration testing"}
  ],
  "relationships": [
    {"source": "V-model", "type": "DEFINES", "target": "verification"},
    {"source": "unit testing", "type": "COMPONENT_OF", "target": "verification"},
    {"source": "integration testing", "type": "COMPONENT_OF", "target": "verification"}
  ]
}

INPUT: "Jama Connect helps automotive and medical device companies manage requirements."
OUTPUT: {
  "entities": [
    {"label": "Tool", "name": "Jama Connect", "properties": {"vendor": "Jama Software"}},
    {"label": "Industry", "name": "Automotive"},
    {"label": "Industry", "name": "Medical Devices"},
    {"label": "Concept", "name": "requirements management"}
  ],
  "relationships": [
    {"source": "Jama Connect", "type": "USED_BY", "target": "Automotive"},
    {"source": "Jama Connect", "type": "USED_BY", "target": "Medical Devices"},
    {"source": "Jama Connect", "type": "RELATED_TO", "target": "requirements management"}
  ]
}

## EXTRACTION GUIDELINES

1. Normalize entity names to lowercase for consistency
2. Preserve acronyms in parentheses: "Natural Language Processing (NLP)"
3. Extract relationships only when explicitly stated or strongly implied
4. Assign confidence scores: 1.0 for explicit mentions, 0.7-0.9 for implied
5. Include "evidence" property on relationships with supporting text snippet

"""

# =============================================================================
# CREATE CUSTOM PROMPT TEMPLATE
# =============================================================================

prompt_template = ERExtractionTemplate(
    template=REQUIREMENTS_DOMAIN_INSTRUCTIONS + ERExtractionTemplate.DEFAULT_TEMPLATE
)
```

### 6.5 Entity Property Assignment

Entity properties are assigned during LLM extraction based on the schema definition and extracted context:

```python
"""How entity properties get populated during extraction."""

# The LLM extracts properties based on:
# 1. Schema definition (what properties exist)
# 2. Context in the source text
# 3. Few-shot examples showing expected output

# Example extraction from text:
# "ISO 26262 is an automotive functional safety standard published by ISO."

# LLM Output:
{
    "entities": [
        {
            "label": "Standard",
            "name": "iso 26262",                    # Normalized
            "properties": {
                "display_name": "ISO 26262",        # Original casing
                "organization": "ISO",              # Extracted from context
                "definition": "automotive functional safety standard"  # Extracted
            }
        },
        {
            "label": "Industry", 
            "name": "automotive",
            "properties": {
                "display_name": "Automotive"
            }
        }
    ],
    "relationships": [
        {
            "source": "iso 26262",
            "target": "automotive",
            "type": "APPLIES_TO",
            "properties": {
                "confidence": 1.0,
                "evidence": "ISO 26262 is an automotive functional safety standard"
            }
        }
    ]
}
```

**Property Population Flow:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROPERTY POPULATION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SCHEMA DEFINES AVAILABLE PROPERTIES                                     │
│     ─────────────────────────────────                                       │
│     NODE_TYPES = [                                                          │
│         {                                                                   │
│             "label": "Standard",                                            │
│             "properties": [                                                 │
│                 {"name": "name", "type": "STRING", "required": True},       │
│                 {"name": "organization", "type": "STRING"},  # Optional     │
│             ]                                                               │
│         }                                                                   │
│     ]                                                                       │
│                                                                             │
│  2. LLM EXTRACTS FROM CONTEXT                                               │
│     ───────────────────────────                                             │
│     Source text: "ISO 26262 is published by the International              │
│                   Organization for Standardization (ISO)."                  │
│                                                                             │
│     LLM identifies:                                                         │
│     - name: "ISO 26262" (explicit)                                          │
│     - organization: "ISO" (extracted from context)                          │
│                                                                             │
│  3. POST-PROCESSING ADDS DERIVED PROPERTIES                                 │
│     ─────────────────────────────────────────                               │
│     - display_name: Original casing preserved                               │
│     - id: Generated UUID                                                    │
│     - confidence: Set based on extraction clarity                           │
│                                                                             │
│  4. NORMALIZATION BEFORE MERGE                                              │
│     ──────────────────────────────                                          │
│     - name: toLower(trim(name))  # For deduplication                        │
│     - All identity properties: Scalar only, never LIST                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 LexicalGraphConfig for Domain-Specific Structure

Use `LexicalGraphConfig` to customize the lexical graph labels to match our domain terminology:

```python
"""Configure lexical graph to use Article/Chunk instead of Document/Chunk."""

from neo4j_graphrag.experimental.components.types import LexicalGraphConfig

# =============================================================================
# CUSTOM LEXICAL GRAPH CONFIGURATION
# =============================================================================
# This maps the generic neo4j_graphrag labels to our domain-specific schema.
#
# Default neo4j_graphrag structure:
#   (:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
#
# Our customized structure:
#   (:Article)-[:HAS_CHUNK]->(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
#
# Note: We keep "Chunk" as-is since it aligns with our schema.
# =============================================================================

lexical_config = LexicalGraphConfig(
    # Node labels
    document_node_label="Article",           # Maps Document → Article
    chunk_node_label="Chunk",                # Keep as Chunk
    
    # Relationship types
    chunk_to_document_relationship_type="FROM_ARTICLE",  # Chunk → Article
    next_chunk_relationship_type="NEXT_CHUNK",           # Sequential linking
    node_to_chunk_relationship_type="MENTIONED_IN",      # Entity → Chunk
    
    # Embedding property
    chunk_embedding_property="embedding",     # Standard embedding property
)

# Use in pipeline:
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=False,  # We're processing HTML
    lexical_graph_config=lexical_config,
    prompt_template=prompt_template,  # Custom extraction prompt
    text_splitter=html_splitter,      # LangChain HTML splitter
)
```

---

## 7. LangChain HTML Integration

### 7.1 LangChain HTML Splitters Overview

LangChain provides three HTML-specific splitters that are directly applicable to our Jama HTML content:

| Splitter | Description | Use Case |
|----------|-------------|----------|
| **HTMLHeaderTextSplitter** | Splits by header tags (h1-h6), preserves hierarchy in metadata | Primary splitter for section-level chunks |
| **HTMLSectionSplitter** | Splits into larger sections, supports XSLT transformation | Converting Avia theme elements to headers |
| **HTMLSemanticPreservingSplitter** | Preserves tables, lists, and structured elements intact | Keeping definition boxes whole |

### 7.2 HTMLHeaderTextSplitter for Jama Content

```python
"""LangChain HTMLHeaderTextSplitter integration for Jama guide content.

This splitter splits HTML by header hierarchy and preserves the header
context in metadata, enabling hierarchical chunk relationships.
"""

from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

# =============================================================================
# HTML HEADER SPLITTER CONFIGURATION
# =============================================================================
# Configure which headers to split on and how to label them in metadata.
# The Jama guide uses h1 for article titles, h2/h3 for sections.
# =============================================================================

headers_to_split_on = [
    ("h1", "article_title"),      # Article-level (Level 0)
    ("h2", "section_heading"),    # Section-level (Level 1)
    ("h3", "subsection_heading"), # Subsection-level (Level 1 or 2)
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False,  # Combine elements under same header
)

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

html_content = """
<html>
<body>
    <h1>Requirements Traceability Guide</h1>
    <p>Introduction to traceability concepts.</p>
    
    <h2>What is Traceability?</h2>
    <p>Traceability is the ability to trace requirements...</p>
    
    <h3>Bidirectional Traceability</h3>
    <p>Forward and backward traceability ensures...</p>
    
    <h2>Benefits of Traceability</h2>
    <p>Key benefits include improved quality...</p>
</body>
</html>
"""

# Split HTML
sections = html_splitter.split_text(html_content)

# Result: List of Document objects with hierarchical metadata
# [
#   Document(
#     page_content="Introduction to traceability concepts.",
#     metadata={"article_title": "Requirements Traceability Guide"}
#   ),
#   Document(
#     page_content="Traceability is the ability to trace requirements...",
#     metadata={
#       "article_title": "Requirements Traceability Guide",
#       "section_heading": "What is Traceability?"
#     }
#   ),
#   Document(
#     page_content="Forward and backward traceability ensures...",
#     metadata={
#       "article_title": "Requirements Traceability Guide",
#       "section_heading": "What is Traceability?",
#       "subsection_heading": "Bidirectional Traceability"
#     }
#   ),
#   ...
# ]
```

### 7.3 HTMLSectionSplitter with XSLT for Avia Theme

The Jama guide uses the Avia theme which has custom elements (message boxes, callouts) that aren't standard headers. We can use XSLT transformation to convert these to splittable headers:

```python
"""HTMLSectionSplitter with custom XSLT for Avia theme elements.

The Avia theme uses custom classes for definition boxes and callouts.
XSLT transformation converts these to standard headers for splitting.
"""

from langchain_text_splitters import HTMLSectionSplitter
from pathlib import Path

# =============================================================================
# CUSTOM XSLT FOR AVIA THEME ELEMENTS
# =============================================================================
# This XSLT converts Avia message boxes to h4 headers so they become
# split boundaries, preserving definitions as separate chunks.
# =============================================================================

AVIA_TO_HEADERS_XSLT = """<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    
    <!-- Identity transform: copy everything by default -->
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>
    
    <!-- Convert Avia message box titles to h4 -->
    <xsl:template match="span[@class='avia_message_box_title']">
        <h4>
            <xsl:apply-templates select="@*|node()"/>
        </h4>
    </xsl:template>
    
    <!-- Convert Avia callout titles to h4 -->
    <xsl:template match="div[contains(@class, 'avia-callout-title')]">
        <h4>
            <xsl:apply-templates select="@*|node()"/>
        </h4>
    </xsl:template>
    
</xsl:stylesheet>
"""

# Save XSLT to file (required by HTMLSectionSplitter)
xslt_path = Path("./data/avia_to_headers.xslt")
xslt_path.parent.mkdir(parents=True, exist_ok=True)
xslt_path.write_text(AVIA_TO_HEADERS_XSLT)

# =============================================================================
# HTML SECTION SPLITTER WITH XSLT
# =============================================================================

headers_to_split_on = [
    ("h1", "article_title"),
    ("h2", "section_heading"),
    ("h3", "subsection_heading"),
    ("h4", "definition_title"),  # Converted from Avia message boxes
]

section_splitter = HTMLSectionSplitter(
    headers_to_split_on=headers_to_split_on,
    xslt_path=str(xslt_path),  # Apply Avia-to-headers transformation
)

# Split HTML with Avia elements
sections = section_splitter.split_text(avia_html_content)
```

### 7.4 LangChainTextSplitterAdapter for neo4j_graphrag

The `LangChainTextSplitterAdapter` bridges LangChain splitters with neo4j_graphrag's `SimpleKGPipeline`:

```python
"""Integrate LangChain HTML splitters with neo4j_graphrag pipeline.

Pattern from genai-graphrag-python/examples/text_splitter_langchain.py
"""

from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# =============================================================================
# TWO-STAGE SPLITTER: HTML Headers → Recursive Character
# =============================================================================
# Stage 1: Split by HTML headers (structure-aware)
# Stage 2: Apply RecursiveCharacterTextSplitter to large sections
# =============================================================================

class HierarchicalHTMLSplitter:
    """Two-stage splitter: HTML headers then character-based for large sections."""
    
    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        large_section_threshold: int = 1500,
    ):
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_element=False,
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self.threshold = large_section_threshold
    
    def split_text(self, html_content: str) -> list:
        """Split HTML: first by headers, then large sections by characters."""
        # Stage 1: HTML header splitting
        header_splits = self.html_splitter.split_text(html_content)
        
        final_splits = []
        for doc in header_splits:
            if len(doc.page_content) > self.threshold:
                # Stage 2: Further split large sections
                sub_splits = self.char_splitter.split_documents([doc])
                final_splits.extend(sub_splits)
            else:
                final_splits.append(doc)
        
        return final_splits


# =============================================================================
# WRAP FOR NEO4J_GRAPHRAG
# =============================================================================

headers_to_split_on = [
    ("h1", "article_title"),
    ("h2", "section_heading"),
    ("h3", "subsection_heading"),
]

hierarchical_splitter = HierarchicalHTMLSplitter(
    headers_to_split_on=headers_to_split_on,
    chunk_size=512,
    chunk_overlap=64,
    large_section_threshold=1500,
)

# Wrap with adapter for neo4j_graphrag compatibility
splitter_adapter = LangChainTextSplitterAdapter(hierarchical_splitter)

# Use in pipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=False,
    text_splitter=splitter_adapter,  # LangChain splitter via adapter
    prompt_template=prompt_template,
    lexical_graph_config=lexical_config,
)
```

### 7.5 Custom HTML DataLoader for Jama Content

Following the pattern from `data_loader_wikipedia.py`, create a custom loader for scraped Jama HTML:

```python
"""Custom DataLoader for Jama HTML content.

Pattern from genai-graphrag-python/examples/data_loader_wikipedia.py
"""

from pathlib import Path
from neo4j_graphrag.experimental.components.pdf_loader import (
    DataLoader,
    DocumentInfo,
    PdfDocument,  # Reused for any document type
)

class JamaHTMLLoader(DataLoader):
    """Load and preprocess Jama guide HTML content."""
    
    def __init__(self, article_index: dict[str, dict] | None = None):
        """Initialize loader with optional article metadata index.
        
        Args:
            article_index: Mapping of article_id to metadata dict
        """
        self.article_index = article_index or {}
    
    async def run(self, filepath: Path) -> PdfDocument:
        """Load HTML file and return as document.
        
        Args:
            filepath: Path to HTML file or article_id
            
        Returns:
            PdfDocument containing preprocessed HTML text
        """
        # Load HTML content
        if isinstance(filepath, str) and not Path(filepath).exists():
            # Treat as article_id, look up in index
            article_meta = self.article_index.get(filepath, {})
            html_path = article_meta.get("html_path")
            if not html_path:
                raise FileNotFoundError(f"Article not found: {filepath}")
            filepath = Path(html_path)
        
        html_content = Path(filepath).read_text(encoding="utf-8")
        
        # Preprocess: extract main content, remove nav/footer
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove non-content elements
        for tag in soup.select("nav, footer, .sidebar, .header, script, style"):
            tag.decompose()
        
        # Extract main content
        main_content = soup.select_one("article, .entry-content, main, .post-content")
        if main_content:
            clean_html = str(main_content)
        else:
            clean_html = str(soup.body) if soup.body else str(soup)
        
        # Build metadata
        article_id = Path(filepath).stem
        metadata = {
            "article_id": article_id,
            "source_path": str(filepath),
            "url": self.article_index.get(article_id, {}).get("url", ""),
        }
        
        return PdfDocument(
            text=clean_html,  # Return HTML for HTMLHeaderTextSplitter
            document_info=DocumentInfo(
                path=str(filepath),
                metadata=metadata,
            ),
        )


# =============================================================================
# USE IN PIPELINE
# =============================================================================

# Build article index from scraper output
article_index = {
    "ch1-art1": {
        "html_path": "./scraped/ch1-art1.html",
        "url": "https://jamasoftware.com/requirements-management-guide/...",
        "title": "What is Requirements Management?",
    },
    # ... more articles
}

html_loader = JamaHTMLLoader(article_index=article_index)

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=NEO4J_DATABASE,
    embedder=embedder,
    from_pdf=False,           # Not processing PDFs
    pdf_loader=html_loader,   # Custom HTML loader (reuses pdf_loader param)
    text_splitter=splitter_adapter,
    prompt_template=prompt_template,
    lexical_graph_config=lexical_config,
)
```

### 7.6 Complete Pipeline Assembly

Putting it all together:

```python
"""Complete Jama ETL Pipeline with LangChain HTML Integration.

Combines:
- Custom HTML DataLoader
- LangChain HTMLHeaderTextSplitter via adapter
- Custom extraction prompt with few-shot examples
- Domain-specific LexicalGraphConfig
"""

import asyncio
from pathlib import Path

from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm import OpenAILLM

from .extraction.schema import get_schema_dict
from .extraction.prompts import REQUIREMENTS_DOMAIN_INSTRUCTIONS
from .loaders.html_loader import JamaHTMLLoader
from .chunking.hierarchical_splitter import HierarchicalHTMLSplitter


def create_jama_kg_pipeline(
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str,
    article_index: dict[str, dict],
) -> SimpleKGPipeline:
    """Create configured KG pipeline for Jama guide processing.
    
    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        article_index: Mapping of article_id to metadata
        
    Returns:
        Configured SimpleKGPipeline
    """
    # Neo4j connection
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )
    driver.verify_connectivity()
    
    # LLM for extraction
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )
    
    # Embeddings
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Custom HTML loader
    html_loader = JamaHTMLLoader(article_index=article_index)
    
    # LangChain HTML splitter with hierarchical chunking
    headers_to_split_on = [
        ("h1", "article_title"),
        ("h2", "section_heading"),
        ("h3", "subsection_heading"),
    ]
    
    hierarchical_splitter = HierarchicalHTMLSplitter(
        headers_to_split_on=headers_to_split_on,
        chunk_size=512,
        chunk_overlap=64,
        large_section_threshold=1500,
    )
    
    splitter_adapter = LangChainTextSplitterAdapter(hierarchical_splitter)
    
    # Custom extraction prompt with domain instructions
    prompt_template = ERExtractionTemplate(
        template=REQUIREMENTS_DOMAIN_INSTRUCTIONS + ERExtractionTemplate.DEFAULT_TEMPLATE
    )
    
    # Domain-specific lexical graph config
    lexical_config = LexicalGraphConfig(
        document_node_label="Article",
        chunk_node_label="Chunk",
        chunk_to_document_relationship_type="FROM_ARTICLE",
        next_chunk_relationship_type="NEXT_CHUNK",
        node_to_chunk_relationship_type="MENTIONED_IN",
        chunk_embedding_property="embedding",
    )
    
    # Create pipeline
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        neo4j_database=neo4j_database,
        embedder=embedder,
        from_pdf=False,
        pdf_loader=html_loader,
        text_splitter=splitter_adapter,
        prompt_template=prompt_template,
        lexical_graph_config=lexical_config,
        schema=get_schema_dict(),
        perform_entity_resolution=True,  # Enable entity resolution
    )


# =============================================================================
# USAGE
# =============================================================================

async def main():
    # Load article index from scraper
    import json
    with open("./output/article_index.json") as f:
        article_index = json.load(f)
    
    # Create pipeline
    pipeline = create_jama_kg_pipeline(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USERNAME"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        neo4j_database=os.getenv("NEO4J_DATABASE"),
        article_index=article_index,
    )
    
    # Process articles
    for article_id in article_index:
        print(f"Processing {article_id}...")
        result = await pipeline.run_async(file_path=article_id)
        print(f"  Extracted: {result.result}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Hierarchical Chunking Strategy

### 8.1 Configuration

```python
"""Hierarchical chunking configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HierarchicalChunkingConfig:
    """Configuration for 3-tier hierarchical chunking.
    
    Level 0: Article Summary
        - One per article
        - Target: ~300 tokens
        - Content: LLM-generated summary or first N chars
    
    Level 1: Section Chunks
        - Natural heading boundaries (h2/h3)
        - Target: 500-1500 tokens
        - No sliding window applied
    
    Level 2: Sliding Window
        - Applied to large sections (> sliding_window_threshold)
        - Overlapping windows for context continuity
        - Each window is child of its section
    """
    
    # Level 0: Summary
    summary_max_tokens: int = 300
    
    # Level 1: Section
    section_min_tokens: int = 50      # Skip very short sections
    section_max_tokens: int = 1500    # Apply sliding window above this
    
    # Level 2: Sliding window
    sliding_window_size: int = 512
    sliding_window_overlap: int = 64
    sliding_window_threshold: int = 1500  # Only apply when section > this
    
    # Tokenizer
    tokenizer_model: str = "cl100k_base"  # GPT-4/ada-002 tokenizer
```

### 8.2 Chunker Implementation

```python
"""Hierarchical 3-tier chunker implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from ..models.content import Article, Section


@dataclass
class HierarchicalChunk:
    """A chunk in the hierarchical structure."""
    
    id: str
    text: str
    level: int                    # 0, 1, or 2
    chunk_type: str               # "summary", "section", "window"
    heading: str | None
    token_count: int
    char_start: int
    char_end: int
    parent_id: str | None = None  # For level 1 → 0, level 2 → 1
    children_ids: list[str] = field(default_factory=list)


class HierarchicalChunker:
    """Create hierarchical 3-tier chunks from articles."""
    
    def __init__(self, config: HierarchicalChunkingConfig | None = None):
        """Initialize chunker.
        
        Args:
            config: Chunking configuration
        """
        from .config import HierarchicalChunkingConfig
        
        self.config = config or HierarchicalChunkingConfig()
        self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._tokenizer.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])
    
    def chunk_article(
        self,
        article: Article,
        summary: str | None = None,
    ) -> list[HierarchicalChunk]:
        """Create hierarchical chunks from an article.
        
        Args:
            article: Article to chunk
            summary: Optional LLM-generated summary (falls back to extraction)
            
        Returns:
            List of HierarchicalChunks at all levels
        """
        chunks: list[HierarchicalChunk] = []
        article_id = article.article_id
        
        # --- Level 0: Article Summary ---
        summary_text = summary or self._extract_summary(article)
        summary_chunk = HierarchicalChunk(
            id=f"{article_id}-L0",
            text=summary_text,
            level=0,
            chunk_type="summary",
            heading=article.title,
            token_count=self.count_tokens(summary_text),
            char_start=0,
            char_end=len(summary_text),
            parent_id=None,
        )
        chunks.append(summary_chunk)
        
        # --- Level 1: Section Chunks ---
        if article.sections:
            char_offset = 0
            
            for i, section in enumerate(article.sections):
                section_text = self._section_to_text(section)
                section_tokens = self.count_tokens(section_text)
                
                # Skip very short sections
                if section_tokens < self.config.section_min_tokens:
                    char_offset += len(section_text)
                    continue
                
                section_chunk = HierarchicalChunk(
                    id=f"{article_id}-L1-{i}",
                    text=section_text,
                    level=1,
                    chunk_type="section",
                    heading=section.heading,
                    token_count=section_tokens,
                    char_start=char_offset,
                    char_end=char_offset + len(section_text),
                    parent_id=summary_chunk.id,
                )
                chunks.append(section_chunk)
                summary_chunk.children_ids.append(section_chunk.id)
                
                # --- Level 2: Sliding Window (if section is large) ---
                if section_tokens > self.config.sliding_window_threshold:
                    window_chunks = self._create_sliding_windows(
                        text=section_text,
                        section_id=section_chunk.id,
                        article_id=article_id,
                        section_index=i,
                        base_char_offset=char_offset,
                    )
                    
                    for wc in window_chunks:
                        chunks.append(wc)
                        section_chunk.children_ids.append(wc.id)
                
                char_offset += len(section_text)
        else:
            # No sections: chunk entire content
            content = article.markdown_content or ""
            content_tokens = self.count_tokens(content)
            
            if content_tokens > self.config.sliding_window_threshold:
                # Large article without sections: create sliding windows
                window_chunks = self._create_sliding_windows(
                    text=content,
                    section_id=summary_chunk.id,
                    article_id=article_id,
                    section_index=0,
                    base_char_offset=0,
                )
                for wc in window_chunks:
                    chunks.append(wc)
                    summary_chunk.children_ids.append(wc.id)
        
        return chunks
    
    def _extract_summary(self, article: Article) -> str:
        """Extract or generate summary from article.
        
        Falls back to first N tokens of content if no summary available.
        """
        if article.summary:
            return self.truncate_to_tokens(
                article.summary, 
                self.config.summary_max_tokens
            )
        
        # Fall back to first N tokens of content
        content = article.markdown_content or ""
        return self.truncate_to_tokens(content, self.config.summary_max_tokens)
    
    def _section_to_text(self, section: Section) -> str:
        """Convert section to text with heading."""
        parts = []
        if section.heading:
            parts.append(f"## {section.heading}")
        parts.append(section.content or "")
        return "\n\n".join(parts)
    
    def _create_sliding_windows(
        self,
        text: str,
        section_id: str,
        article_id: str,
        section_index: int,
        base_char_offset: int,
    ) -> list[HierarchicalChunk]:
        """Create overlapping sliding window chunks.
        
        Args:
            text: Text to split
            section_id: Parent section chunk ID
            article_id: Article ID for chunk ID generation
            section_index: Section index for chunk ID
            base_char_offset: Character offset in original article
            
        Returns:
            List of Level 2 chunks
        """
        chunks = []
        
        # Tokenize for accurate splitting
        tokens = self._tokenizer.encode(text)
        
        window_size = self.config.sliding_window_size
        overlap = self.config.sliding_window_overlap
        step = window_size - overlap
        
        window_index = 0
        start_token = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + window_size, len(tokens))
            window_tokens = tokens[start_token:end_token]
            window_text = self._tokenizer.decode(window_tokens)
            
            # Calculate character positions
            # This is approximate; for exact positions, we'd need to track carefully
            char_start = base_char_offset + len(
                self._tokenizer.decode(tokens[:start_token])
            )
            char_end = char_start + len(window_text)
            
            chunk = HierarchicalChunk(
                id=f"{article_id}-L1-{section_index}-L2-{window_index}",
                text=window_text,
                level=2,
                chunk_type="window",
                heading=None,
                token_count=len(window_tokens),
                char_start=char_start,
                char_end=char_end,
                parent_id=section_id,
            )
            chunks.append(chunk)
            
            start_token += step
            window_index += 1
            
            # Stop if we've covered all content
            if end_token >= len(tokens):
                break
        
        return chunks
```

---

## 9. Resource Extraction

### 9.1 Definition Block Extraction

Based on the HTML sample provided:

```python
"""Definition block extraction from Avia message boxes."""

import re
from bs4 import BeautifulSoup, Tag

from ..models.resource import DefinitionResource


def extract_definitions(content: Tag, base_url: str) -> list[DefinitionResource]:
    """Extract definition blocks from Avia message boxes.
    
    Target HTML pattern:
    <div class="avia_message_box ...">
        <span class="avia_message_box_title">DEFINITION OF TERM:</span>
        <div class="avia_message_box_content">
            <p><strong>Term</strong> is definition text...</p>
        </div>
    </div>
    
    Args:
        content: BeautifulSoup Tag containing article content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of DefinitionResource objects
    """
    definitions = []
    
    # Find Avia message boxes
    for box in content.select(".avia_message_box"):
        # Check for green color class (definition style)
        classes = " ".join(box.get("class", []))
        if "avia-color-green" not in classes:
            continue  # Not a definition box
        
        # Extract term from title
        title_elem = box.select_one(".avia_message_box_title")
        if not title_elem:
            continue
        
        title_text = title_elem.get_text(strip=True)
        
        # Parse "DEFINITION OF X:" pattern
        match = re.match(
            r"DEFINITION\s+OF\s+(.+?):",
            title_text,
            re.IGNORECASE
        )
        if not match:
            continue
        
        term = match.group(1).strip()
        
        # Extract definition content
        content_elem = box.select_one(".avia_message_box_content")
        if not content_elem:
            continue
        
        definition_text = content_elem.get_text(strip=True)
        
        # Check for link to glossary
        glossary_link = content_elem.select_one('a[href*="glossary"]')
        url = None
        if glossary_link:
            from urllib.parse import urljoin
            url = urljoin(base_url, glossary_link.get("href", ""))
        
        definitions.append(DefinitionResource(
            term=term,
            definition_text=definition_text,
            url=url
        ))
    
    return definitions
```

---

## 10. Industry Taxonomy

### 9.1 Taxonomy Definition

```python
"""Industry taxonomy for mapping extracted variants to canonical names.

Maps 96 extracted industry variants to 19 canonical industries.
"""

from __future__ import annotations

from dataclasses import dataclass
from rapidfuzz import fuzz


# Canonical industry taxonomy with variant mappings
INDUSTRY_TAXONOMY: dict[str, list[str]] = {
    "Aerospace & Defense": [
        "aerospace",
        "aerospace and defense",
        "aerospace & defense",
        "defense",
        "defense contracting",
        "uk defense",
        "aviation",
        "aircraft",
        "airplanes",
        "aero engines",
        "space systems",
        "spacecraft",
        "civil airborne systems",
        "military systems",
        "defense industry",
    ],
    "Automotive": [
        "automotive",
        "automotive industry",
        "automobiles",
        "automotive manufacturing",
        "automotive technology",
        "automotive development",
        "automotive testing",
        "autonomous vehicles",
        "autonomous vehicle development",
        "electric vehicles",
        "vehicle systems",
    ],
    "Medical Devices": [
        "medical device",
        "medical devices",
        "medical",
        "medical industry",
        "medical device industry",
        "medical device & life sciences",
        "medical device development",
        "medical device sector",
        "medical device software",
        "medical technology",
        "medtech",
        "healthcare devices",
    ],
    "Pharmaceuticals & Life Sciences": [
        "pharmaceutical",
        "pharmaceuticals",
        "pharma",
        "life sciences",
        "biotech",
        "biotechnology",
        "drug development",
        "clinical trials",
    ],
    "Semiconductor & Electronics": [
        "semiconductor",
        "semiconductor industry",
        "semiconductor design",
        "semiconductor manufacturing",
        "semiconductor technology",
        "semiconductor companies",
        "electronics",
        "consumer electronics",
        "electronic systems",
        "chip design",
    ],
    "Financial Services": [
        "financial services",
        "banking",
        "finance",
        "insurance",
        "fintech",
        "banking and finance",
        "financial industry",
    ],
    "Telecommunications": [
        "telecommunications",
        "telecom",
        "communications",
        "wireless",
        "networking",
        "5g",
    ],
    "Energy & Utilities": [
        "energy",
        "utilities",
        "power generation",
        "oil and gas",
        "oil & gas",
        "renewable energy",
        "nuclear energy",
        "smart grid",
    ],
    "Industrial & Manufacturing": [
        "industrial",
        "manufacturing",
        "industrial automation",
        "factory automation",
        "industrial equipment",
        "machinery",
        "heavy equipment",
    ],
    "Rail & Transportation": [
        "rail",
        "railway",
        "transportation",
        "transit",
        "locomotives",
        "train systems",
        "mass transit",
    ],
    "Government & Public Sector": [
        "government",
        "public sector",
        "federal",
        "state government",
        "local government",
        "public agencies",
    ],
    "Consumer Products": [
        "consumer products",
        "consumer goods",
        "consumer electronics",
        "retail",
        "cpg",
    ],
    "Software & Technology": [
        "software",
        "technology",
        "it",
        "information technology",
        "tech industry",
        "saas",
        "enterprise software",
    ],
    "Construction & Infrastructure": [
        "construction",
        "infrastructure",
        "building",
        "civil engineering",
        "architecture",
    ],
    "Agriculture & Food": [
        "agriculture",
        "food",
        "agtech",
        "food processing",
        "farming",
    ],
    "Marine & Shipbuilding": [
        "marine",
        "shipbuilding",
        "maritime",
        "naval",
        "offshore",
    ],
    "Nuclear": [
        "nuclear",
        "nuclear power",
        "nuclear industry",
        "nuclear safety",
    ],
    "Space": [
        "space",
        "space industry",
        "satellite",
        "space exploration",
    ],
    "Education & Research": [
        "education",
        "research",
        "academia",
        "universities",
        "r&d",
    ],
}


# Technology concepts that should NOT be classified as Industry
TECHNOLOGY_CONCEPTS = {
    "artificial intelligence",
    "ai",
    "machine learning",
    "ml",
    "cloud computing",
    "high-performance computing",
    "internet of things",
    "iot",
    "cyber-physical systems",
    "automation",
    "software development",
    "software applications",
    "software technologies",
    "digital transformation",
    "embedded systems",
    "data analytics",
    "big data",
}


@dataclass
class IndustryClassificationResult:
    """Result of industry classification."""
    
    canonical_name: str | None      # None if should be deleted
    original_name: str
    confidence: float
    should_reclassify_to: str | None  # e.g., "Concept" if misclassified


class IndustryClassifier:
    """Classify extracted industry strings to canonical names."""
    
    def __init__(
        self,
        taxonomy: dict[str, list[str]] | None = None,
        technology_concepts: set[str] | None = None,
        fuzzy_threshold: float = 85.0,
    ):
        """Initialize classifier.
        
        Args:
            taxonomy: Industry taxonomy mapping
            technology_concepts: Set of technology terms to reclassify
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.taxonomy = taxonomy or INDUSTRY_TAXONOMY
        self.technology_concepts = technology_concepts or TECHNOLOGY_CONCEPTS
        self.fuzzy_threshold = fuzzy_threshold
        
        # Build reverse lookup: variant → canonical
        self._variant_to_canonical: dict[str, str] = {}
        for canonical, variants in self.taxonomy.items():
            for variant in variants:
                self._variant_to_canonical[variant.lower()] = canonical
    
    def classify(self, industry_name: str) -> IndustryClassificationResult:
        """Classify an industry name to canonical form.
        
        Args:
            industry_name: Extracted industry name
            
        Returns:
            Classification result with canonical name and confidence
        """
        normalized = industry_name.lower().strip()
        
        # Check if it's actually a technology concept
        if normalized in self.technology_concepts:
            return IndustryClassificationResult(
                canonical_name=None,
                original_name=industry_name,
                confidence=1.0,
                should_reclassify_to="Concept",
            )
        
        # Exact match in taxonomy
        if normalized in self._variant_to_canonical:
            return IndustryClassificationResult(
                canonical_name=self._variant_to_canonical[normalized],
                original_name=industry_name,
                confidence=1.0,
                should_reclassify_to=None,
            )
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        
        for variant, canonical in self._variant_to_canonical.items():
            score = fuzz.ratio(normalized, variant)
            if score > best_score:
                best_score = score
                best_match = canonical
        
        if best_score >= self.fuzzy_threshold:
            return IndustryClassificationResult(
                canonical_name=best_match,
                original_name=industry_name,
                confidence=best_score / 100.0,
                should_reclassify_to=None,
            )
        
        # No match found - may be a new industry or invalid
        return IndustryClassificationResult(
            canonical_name=None,
            original_name=industry_name,
            confidence=0.0,
            should_reclassify_to=None,  # Requires manual review
        )
    
    def classify_batch(
        self, 
        industry_names: list[str]
    ) -> dict[str, IndustryClassificationResult]:
        """Classify multiple industry names.
        
        Args:
            industry_names: List of extracted industry names
            
        Returns:
            Mapping of original name to classification result
        """
        return {name: self.classify(name) for name in industry_names}
```

---

## 10. Implementation Phases

### Phase 1: Project Setup (Day 1)

- [ ] Create new branch: `refactor/neo4j-graphrag-migration`
- [ ] Update `pyproject.toml` with new dependencies
- [ ] Create new directory structure
- [ ] Set up logging with structlog
- [ ] Create configuration management

### Phase 2: Models (Days 2-3)

- [ ] Refactor models into `models/` directory
- [ ] Implement `BaseResource` and all resource types
- [ ] Add `display_name` to entity models
- [ ] Add normalization utilities
- [ ] Write model tests

### Phase 3: Resource Extraction (Days 4-5)

- [ ] Extend parser with `ResourceExtractor`
- [ ] Implement image extraction (content only)
- [ ] Implement definition block extraction
- [ ] Implement webinar/video extraction
- [ ] Implement external link extraction
- [ ] Write extraction tests

### Phase 4: Hierarchical Chunking (Days 6-7)

- [ ] Implement `HierarchicalChunker`
- [ ] Create Level 0 (summary) chunks
- [ ] Create Level 1 (section) chunks
- [ ] Create Level 2 (sliding window) chunks
- [ ] Add parent-child relationships
- [ ] Write chunking tests

### Phase 5: neo4j_graphrag Integration (Days 8-10)

- [ ] Define schema (NODE_TYPES, RELATIONSHIP_TYPES, PATTERNS)
- [ ] Create `JamaKGPipeline` wrapper
- [ ] Test extraction on sample articles
- [ ] Tune prompts for domain accuracy
- [ ] Write extraction tests

### Phase 6: Post-Processing (Days 11-12)

- [ ] Implement `IndustryClassifier`
- [ ] Implement entity type validation
- [ ] Implement glossary auto-linking
- [ ] Implement entity name normalization
- [ ] Write post-processing tests

### Phase 7: Graph Writing (Days 13-14)

- [ ] Implement supplementary Cypher execution
- [ ] Create constraint and index setup
- [ ] Implement batch transaction handling
- [ ] Add embedding integration
- [ ] Write graph writing tests

### Phase 8: Validation & CLI (Days 15-16)

- [ ] Implement validation queries
- [ ] Create validation reporter
- [ ] Update CLI with new pipeline
- [ ] Add progress reporting
- [ ] Write integration tests

### Phase 9: Testing & Documentation (Days 17-18)

- [ ] Run full pipeline on sample data
- [ ] Validate output quality
- [ ] Fix any issues found
- [ ] Update README and CLAUDE.md
- [ ] Document schema for MCP server update

### Phase 10: Production Run (Day 19-20)

- [ ] Run full pipeline on production data
- [ ] Validate against quality metrics
- [ ] Load to Neo4j AuraDB
- [ ] Run validation queries
- [ ] Document results

---

## 11. Testing Requirements

### 11.1 Unit Tests

```python
# tests/test_normalization.py

def test_normalize_entity_name():
    """Test entity name normalization."""
    from jama_scraper.postprocessing.normalizer import normalize_entity_name
    
    assert normalize_entity_name("Artificial Intelligence (AI)") == "artificial intelligence (ai)"
    assert normalize_entity_name("  Traceability  ") == "traceability"
    assert normalize_entity_name("V&V") == "v&v"
    assert normalize_entity_name("Benefit-Risk Analysis") == "benefit-risk analysis"
    assert normalize_entity_name("ISO\u00a013485") == "iso 13485"  # Non-breaking space


def test_normalize_empty_string():
    """Test normalization of empty/None values."""
    from jama_scraper.postprocessing.normalizer import normalize_entity_name
    
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("   ") == ""


# tests/test_industry_taxonomy.py

def test_industry_exact_match():
    """Test exact taxonomy matching."""
    from jama_scraper.postprocessing.industry_taxonomy import IndustryClassifier
    
    classifier = IndustryClassifier()
    
    result = classifier.classify("automotive industry")
    assert result.canonical_name == "Automotive"
    assert result.confidence == 1.0


def test_industry_fuzzy_match():
    """Test fuzzy taxonomy matching."""
    from jama_scraper.postprocessing.industry_taxonomy import IndustryClassifier
    
    classifier = IndustryClassifier()
    
    result = classifier.classify("automative")  # Typo
    assert result.canonical_name == "Automotive"
    assert result.confidence >= 0.85


def test_industry_reclassification():
    """Test technology concept reclassification."""
    from jama_scraper.postprocessing.industry_taxonomy import IndustryClassifier
    
    classifier = IndustryClassifier()
    
    result = classifier.classify("artificial intelligence")
    assert result.canonical_name is None
    assert result.should_reclassify_to == "Concept"
    
    result = classifier.classify("machine learning")
    assert result.should_reclassify_to == "Concept"


# tests/test_chunking.py

def test_hierarchical_chunking_levels():
    """Test that chunks are created at all levels."""
    from jama_scraper.chunking.hierarchical_chunker import HierarchicalChunker
    from jama_scraper.models.content import Article, Section
    
    article = Article(
        article_id="test-art",
        title="Test Article",
        url="http://example.com/test",
        markdown_content="Long content here...",
        sections=[
            Section(heading="Section 1", content="Content " * 500),
            Section(heading="Section 2", content="Content " * 100),
        ],
    )
    
    chunker = HierarchicalChunker()
    chunks = chunker.chunk_article(article)
    
    # Should have L0, L1, and possibly L2 chunks
    levels = {c.level for c in chunks}
    assert 0 in levels  # Summary
    assert 1 in levels  # Sections


def test_chunk_parent_child_relationships():
    """Test hierarchical relationships are set correctly."""
    from jama_scraper.chunking.hierarchical_chunker import HierarchicalChunker
    from jama_scraper.models.content import Article, Section
    
    article = Article(
        article_id="test-art",
        title="Test Article",
        url="http://example.com/test",
        sections=[
            Section(heading="Section 1", content="Content " * 500),
        ],
    )
    
    chunker = HierarchicalChunker()
    chunks = chunker.chunk_article(article)
    
    l0_chunk = next(c for c in chunks if c.level == 0)
    l1_chunks = [c for c in chunks if c.level == 1]
    
    for l1 in l1_chunks:
        assert l1.parent_id == l0_chunk.id
        assert l1.id in l0_chunk.children_ids


# tests/test_resource_extraction.py

def test_extract_definition_blocks():
    """Test definition block extraction from Avia message boxes."""
    from bs4 import BeautifulSoup
    from jama_scraper.parser import extract_definitions
    
    html = '''
    <div class="avia_message_box avia-color-green">
        <span class="avia_message_box_title">DEFINITION OF NLP:</span>
        <div class="avia_message_box_content">
            <p><strong>Natural Language Processing (NLP)</strong> is a branch of AI...</p>
        </div>
    </div>
    '''
    
    soup = BeautifulSoup(html, "html.parser")
    definitions = extract_definitions(soup, "http://example.com")
    
    assert len(definitions) == 1
    assert definitions[0].term == "NLP"
    assert "Natural Language Processing" in definitions[0].definition_text


def test_filter_decorative_images():
    """Test that decorative images are filtered out."""
    from jama_scraper.parser import ResourceExtractor
    
    extractor = ResourceExtractor("http://example.com")
    
    # This should be filtered
    assert extractor._is_decorative_image(
        url="http://example.com/logo.png",
        alt_text="Company Logo",
        img=None  # Mock tag
    )
    
    # This should NOT be filtered
    assert not extractor._is_decorative_image(
        url="http://example.com/diagram.png",
        alt_text="Requirements traceability matrix diagram",
        img=None
    )
```

### 11.2 Integration Tests

```python
# tests/test_integration.py

import pytest
from neo4j import GraphDatabase


@pytest.fixture
def neo4j_driver():
    """Create Neo4j driver for testing."""
    import os
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687"),
        auth=("neo4j", os.getenv("NEO4J_TEST_PASSWORD", "password")),
    )
    yield driver
    driver.close()


def test_no_duplicate_entities(neo4j_driver):
    """Verify no case-variant duplicates after load."""
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (e:Entity)
            WITH toLower(trim(e.name)) as normalized, count(e) as cnt
            WHERE cnt > 1
            RETURN count(*) as duplicate_groups
        """)
        
        record = result.single()
        assert record["duplicate_groups"] == 0, "Found case-variant duplicates"


def test_industry_consolidation(neo4j_driver):
    """Verify industries are consolidated to taxonomy."""
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (i:Industry)
            RETURN count(i) as industry_count
        """)
        
        record = result.single()
        assert record["industry_count"] <= 19, "Too many industry nodes"


def test_no_list_properties(neo4j_driver):
    """Verify no accidental LIST properties on identity fields."""
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (n)
            WHERE n.name IS NOT NULL AND n.name IS :: LIST<ANY>
            RETURN count(n) as list_count
        """)
        
        record = result.single()
        assert record["list_count"] == 0, "Found LIST-type name properties"


def test_chunk_content_stored(neo4j_driver):
    """Verify chunks have actual text content."""
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.text IS NULL OR c.text = ''
            RETURN count(c) as empty_chunks
        """)
        
        record = result.single()
        assert record["empty_chunks"] == 0, "Found chunks without content"


def test_hierarchical_chunk_structure(neo4j_driver):
    """Verify chunk hierarchy is correct."""
    with neo4j_driver.session() as session:
        # All L1 chunks should have L0 parent
        result = session.run("""
            MATCH (c:Chunk {level: 1})
            WHERE NOT (c)-[:CHILD_OF]->(:Chunk {level: 0})
            RETURN count(c) as orphan_l1
        """)
        
        record = result.single()
        assert record["orphan_l1"] == 0, "Found L1 chunks without L0 parent"
        
        # All L2 chunks should have L1 parent
        result = session.run("""
            MATCH (c:Chunk {level: 2})
            WHERE NOT (c)-[:CHILD_OF]->(:Chunk {level: 1})
            RETURN count(c) as orphan_l2
        """)
        
        record = result.single()
        assert record["orphan_l2"] == 0, "Found L2 chunks without L1 parent"
```

---

## 12. Validation Queries

### 12.1 Post-Load Validation

```cypher
// =============================================================================
// VALIDATION QUERY 1: Check for case-variant duplicates
// Expected: 0 rows
// =============================================================================
MATCH (e:Entity)
WHERE e.name IS NOT NULL
WITH toLower(trim(e.name)) as normalized, collect(e) as entities
WHERE size(entities) > 1
RETURN normalized, 
       size(entities) as duplicate_count,
       [e IN entities | e.display_name] as variants
ORDER BY duplicate_count DESC;


// =============================================================================
// VALIDATION QUERY 2: Verify industry consolidation
// Expected: ≤19 industries, all matching taxonomy
// =============================================================================
MATCH (i:Industry)
RETURN i.name as normalized_name, 
       i.display_name as display_name,
       count{(i)<-[:MENTIONS]-()} as mention_count
ORDER BY i.name;


// =============================================================================
// VALIDATION QUERY 3: Check for LIST-type properties
// Expected: 0 rows
// =============================================================================
MATCH (n)
WHERE n.name IS NOT NULL AND n.name IS :: LIST<ANY>
RETURN labels(n) as labels, n.name as problematic_value;

MATCH (n)
WHERE n.id IS NOT NULL AND n.id IS :: LIST<ANY>
RETURN labels(n) as labels, n.id as problematic_value;


// =============================================================================
// VALIDATION QUERY 4: Verify chunk content storage
// Expected: 0 rows (all chunks should have text)
// =============================================================================
MATCH (c:Chunk)
WHERE c.text IS NULL OR trim(c.text) = ''
RETURN c.id as chunk_id, c.level as level, c.chunk_type as type;


// =============================================================================
// VALIDATION QUERY 5: Verify hierarchical chunk structure
// Expected: Proper parent-child relationships
// =============================================================================
// L0 chunks should have no parent
MATCH (c:Chunk {level: 0})
WHERE (c)-[:CHILD_OF]->()
RETURN c.id as invalid_l0_chunk;

// L1 chunks should have L0 parent
MATCH (c:Chunk {level: 1})
WHERE NOT (c)-[:CHILD_OF]->(:Chunk {level: 0})
RETURN c.id as orphan_l1_chunk;

// L2 chunks should have L1 parent
MATCH (c:Chunk {level: 2})
WHERE NOT (c)-[:CHILD_OF]->(:Chunk {level: 1})
RETURN c.id as orphan_l2_chunk;


// =============================================================================
// VALIDATION QUERY 6: Check orphaned entities
// Expected: Review any entities with no relationships
// =============================================================================
MATCH (e:Entity)
WHERE NOT (e)-[]-()
RETURN labels(e) as labels, e.name as name, e.display_name as display_name
ORDER BY labels, name;


// =============================================================================
// VALIDATION QUERY 7: Verify glossary term linking
// =============================================================================
MATCH (g:GlossaryTerm)
OPTIONAL MATCH (g)<-[r]-()
WITH g, count(r) as rel_count
RETURN 
    count(g) as total_terms,
    sum(CASE WHEN rel_count > 0 THEN 1 ELSE 0 END) as linked_terms,
    sum(CASE WHEN rel_count = 0 THEN 1 ELSE 0 END) as unlinked_terms;


// =============================================================================
// VALIDATION QUERY 8: Entity type distribution
// =============================================================================
MATCH (e:Entity)
WITH labels(e) as entity_labels, count(e) as cnt
RETURN entity_labels, cnt
ORDER BY cnt DESC;


// =============================================================================
// VALIDATION QUERY 9: Relationship counts by type
// =============================================================================
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC;


// =============================================================================
// VALIDATION QUERY 10: Resource counts by type
// =============================================================================
MATCH (r:Resource)
RETURN r.resource_type as type, count(r) as count
ORDER BY count DESC;


// =============================================================================
// VALIDATION QUERY 11: Verify entity name vs display_name
// Expected: All entities should have both
// =============================================================================
MATCH (e:Entity)
WHERE e.display_name IS NULL
RETURN e.name as name_missing_display, labels(e) as labels
LIMIT 10;

MATCH (e:Entity)
WHERE e.name <> toLower(trim(e.name))
RETURN e.name as unnormalized_name, e.display_name as display_name
LIMIT 10;
```

### 12.2 Quality Metrics

After running validation queries, document these metrics:

| Metric | Target | Actual |
|--------|--------|--------|
| Case-variant duplicate groups | 0 | |
| Industry nodes | ≤19 | |
| LIST-type properties | 0 | |
| Chunks without content | 0 | |
| Orphan L1 chunks | 0 | |
| Orphan L2 chunks | 0 | |
| Total entities | ~3,000-4,000 | |
| Entity-to-entity relationships | >500 | |
| Linked glossary terms | >100 | |

---

## Appendix A: Environment Variables

```bash
# .env.example

# Neo4j AuraDB Connection
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_DATABASE=neo4j

# OpenAI API
OPENAI_API_KEY=sk-your-key-here

# Optional: Anthropic for extraction (if using Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Logging
LOG_LEVEL=INFO
```

---

## Appendix B: CLI Usage

```bash
# Full pipeline run
jama-scrape --output ./output --load-neo4j

# Scrape only (no graph loading)
jama-scrape --output ./output --scrape-only

# Resume from checkpoint
jama-scrape --output ./output --resume

# Estimate embedding cost
jama-scrape --output ./output --estimate-cost

# Validate existing database
jama-scrape --validate-only

# Debug mode with verbose logging
jama-scrape --output ./output --debug
```

---

## Appendix C: Schema Migration Notes

When updating the MCP server (separate phase), these schema changes will require query updates:

| Change | MCP Server Impact |
|--------|-------------------|
| `name` normalized, `display_name` for UI | Update display queries to use `display_name` |
| Chunk `text` property (was missing) | Update RAG queries to return `text` |
| Chunk `level` property | Can use for hierarchical retrieval |
| `[:CHILD_OF]` chunk relationships | Enable "expand context" features |
| `[:RELATED_TO]` article relationships | Enable "related articles" features |
| Resource nodes | New queries for resource retrieval |
| Entity provenance as relationships | Update "where mentioned" queries |

---

*Document Version: 1.0*
*Created: January 2025*
*Author: Claude (Anthropic)*
