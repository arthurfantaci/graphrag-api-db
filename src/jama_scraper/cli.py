"""Command-line interface for the Jama Guide Scraper.

This CLI runs the complete Neo4j pipeline:
1. Scrape all articles and glossary
2. Extract entities and relationships (LangExtract)
3. Chunk articles for RAG retrieval
4. Generate embeddings for vector search
5. Export Neo4j import files (CSV + Cypher)
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .exceptions import BrowserNotInstalledError, PlaywrightNotAvailableError
from .scraper import run_scraper

console = Console()


def main() -> None:
    """Run the Jama Guide Scraper CLI.

    Executes the complete pipeline for Neo4j graph database with vector index.
    """
    # Load .env file for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Scrape Jama's Requirements Management Guide for Neo4j import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool runs the complete pipeline to generate Neo4j import files:

  1. Scrape all 103 articles + glossary from Jama's guide
  2. Extract entities and relationships using LangExtract
  3. Chunk articles into RAG-friendly segments (3-tier)
  4. Generate embeddings for Neo4j vector index
  5. Export CSV and Cypher files for Neo4j import

Required environment variables:
  ANTHROPIC_API_KEY  - For entity extraction (LangExtract)
  OPENAI_API_KEY     - For embeddings (text-embedding-3-small)

Examples:
  # Run full pipeline (default output: ./output)
  jama-scrape

  # Specify output directory
  jama-scrape -o ./neo4j-data

  # Resume from checkpoint (if interrupted)
  jama-scrape --resume

  # Estimate embedding cost before running
  jama-scrape --estimate-cost
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for Neo4j files (default: ./output)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skips already processed articles)",
    )

    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate embedding cost and exit (dry run)",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Use headless browser for JavaScript-rendered content. "
            "Requires: uv sync --group browser && playwright install chromium"
        ),
    )

    args = parser.parse_args()

    console.print("[bold cyan]Jama Guide → Neo4j Pipeline[/]")
    console.print(f"Output directory: {args.output}")
    console.print()
    console.print("Pipeline stages:")
    console.print("  1. Scrape articles + glossary")
    console.print("  2. Extract entities/relationships")
    console.print("  3. Chunk for RAG retrieval")
    console.print("  4. Generate embeddings")
    console.print("  5. Export Neo4j files")
    console.print()

    try:
        asyncio.run(
            run_scraper(
                output_dir=args.output,
                resume_enrichment=args.resume,
                estimate_cost=args.estimate_cost,
                use_browser=args.browser,
            )
        )
    except PlaywrightNotAvailableError:
        console.print("\n[red]Error: Playwright not installed[/]")
        console.print("Install with: [cyan]uv sync --group browser[/]")
        raise SystemExit(1) from None
    except BrowserNotInstalledError:
        console.print("\n[red]Error: Browser binaries not installed[/]")
        console.print("Install with: [cyan]playwright install chromium[/]")
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/]")
        console.print("Run with [cyan]--resume[/] to continue from checkpoint")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
