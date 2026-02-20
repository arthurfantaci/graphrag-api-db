# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-01-20

### Added

- Async web scraping pipeline with `httpx` and optional Playwright for JS-rendered content
- Neo4j GraphRAG integration via `neo4j_graphrag.SimpleKGPipeline`
- Schema-constrained entity extraction with 10 node types and 10 relationship types
- LangChain `HTMLHeaderTextSplitter` for hierarchical document chunking
- Optional Chonkie `SemanticChunker` with Savitzky-Golay boundary detection
- Voyage AI `voyage-4` embeddings (1024d) with OpenAI fallback
- Industry taxonomy normalization consolidating 100+ variants into 18 canonical industries
- Entity post-processing pipeline: normalize, deduplicate, cleanup, consolidate, backfill, summarize
- LangExtract augmentation with source grounding (text span provenance)
- Leiden community detection with LLM-generated community summaries
- Supplementary graph structure: Chapter, Resource (Image/Video/Webinar), and Glossary nodes
- Comprehensive validation framework with pass/fail checks and repair operations
- CLI with `scrape` and `validate` subcommands, dry-run support, and cost estimation
- Pre-flight validation before pipeline ingestion
- CI/CD pipeline with linting (Ruff), type checking (ty), unit tests, and integration tests

### Fixed

- Cypher double-WHERE syntax in relabel query (use WITH bridge)
- Chunk ordering property name (`chunk_index` → `index`)
- Voyage AI dimensions in `.env.example` (1536d → 1024d)
