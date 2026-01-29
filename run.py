#!/usr/bin/env python3
"""Quick-run script for the GraphRAG Knowledge Graph Pipeline.

This runs the complete 5-stage pipeline:
1. Scrape all articles and glossary
2. Process through SimpleKGPipeline (chunking, extraction, embeddings)
3. Apply entity normalization and industry consolidation
4. Create supplementary structure (chapters, resources, glossary)
5. Run validation checks (optional)

Usage:
    python run.py

    # Or with UV:
    uv run python run.py

    # Scrape only (no Neo4j):
    SCRAPE_ONLY=1 python run.py
"""

import asyncio
import os
from pathlib import Path

# Add src to path for development
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphrag_kg_pipeline import run_scraper


async def main():
    """Run the pipeline with default settings."""
    scrape_only = os.getenv("SCRAPE_ONLY", "").lower() in ("1", "true", "yes")

    guide = await run_scraper(
        output_dir=Path("output"),
        scrape_only=scrape_only,
        run_validation=not scrape_only,
    )

    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Title: {guide.metadata.title}")
    print(f"Chapters: {len(guide.chapters)}")
    print(f"Total Articles: {guide.total_articles}")
    print(f"Total Words: {guide.total_word_count:,}")
    if guide.glossary:
        print(f"Glossary Terms: {guide.glossary.term_count}")
    if scrape_only:
        print("\nOutput files saved to: ./output/")
    else:
        print("\nData loaded to Neo4j knowledge graph")


if __name__ == "__main__":
    asyncio.run(main())
