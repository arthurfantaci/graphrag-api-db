#!/usr/bin/env python3
"""Quick-run script for the Jama Guide Scraper.

Usage:
    python run.py

    # Or with UV:
    uv run python run.py
"""

import asyncio
from pathlib import Path

# Add src to path for development
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphrag_kg_pipeline import run_scraper


async def main():
    """Run the scraper with default settings."""
    guide = await run_scraper(
        output_dir=Path("output"),
        formats=["json", "jsonl", "markdown"],
    )

    print(f"\n{'=' * 60}")
    print("SCRAPING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Title: {guide.metadata.title}")
    print(f"Chapters: {len(guide.chapters)}")
    print(f"Total Articles: {guide.total_articles}")
    print(f"Total Words: {guide.total_word_count:,}")
    if guide.glossary:
        print(f"Glossary Terms: {guide.glossary.term_count}")
    print("\nOutput files saved to: ./output/")


if __name__ == "__main__":
    asyncio.run(main())
