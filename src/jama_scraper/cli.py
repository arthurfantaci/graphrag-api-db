"""Command-line interface for the Jama Guide Scraper.

This CLI runs the complete Neo4j pipeline:
1. Scrape all articles and glossary
2. Extract entities and relationships (LangExtract)
3. Chunk articles for RAG retrieval
4. Generate embeddings for vector search
5. Export Neo4j import files (CSV + Cypher)
6. [Optional] Load data directly into Neo4j database
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .exceptions import (
    BrowserNotInstalledError,
    Neo4jConfigError,
    PlaywrightNotAvailableError,
)
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
  6. [Optional] Load data directly into Neo4j database

Required environment variables:
  OPENAI_API_KEY     - For entity extraction + embeddings

Optional environment variables (for --load-neo4j):
  NEO4J_URI          - Database URI (e.g., neo4j+s://xxx.databases.neo4j.io)
  NEO4J_USERNAME     - Database username (default: neo4j)
  NEO4J_PASSWORD     - Database password

Examples:
  # Run full pipeline (default output: ./output)
  jama-scrape

  # Specify output directory
  jama-scrape -o ./neo4j-data

  # Resume from checkpoint (if interrupted)
  jama-scrape --resume

  # Load data directly into Neo4j database
  jama-scrape --load-neo4j

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

    parser.add_argument(
        "--load-neo4j",
        action="store_true",
        help=(
            "Load data directly into Neo4j database after export. "
            "Requires: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environment variables."
        ),
    )

    parser.add_argument(
        "--load-only",
        action="store_true",
        help=(
            "Skip pipeline stages 1-5, only load existing export into Neo4j. "
            "Use after a previous run to reload data without re-scraping."
        ),
    )

    args = parser.parse_args()

    console.print("[bold cyan]Jama Guide → Neo4j Pipeline[/]")
    console.print(f"Output directory: {args.output}")
    console.print()

    # Handle --load-only: skip pipeline, just load into Neo4j
    if args.load_only:
        console.print("Pipeline stage: Load existing export into Neo4j")
        console.print()
        try:
            from .neo4j_loader import Neo4jLoader, get_neo4j_config

            config = get_neo4j_config()
            if config is None:
                raise Neo4jConfigError
            uri, username, password = config
            loader = Neo4jLoader(uri=uri, username=username, password=password)
            try:
                loader.load_all(args.output)
            finally:
                loader.close()
            return
        except Neo4jConfigError:
            console.print("\n[red]Error: Neo4j configuration missing[/]")
            console.print("Set: [cyan]NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD[/]")
            raise SystemExit(1) from None
        except FileNotFoundError as e:
            console.print(f"\n[red]Error: {e}[/]")
            console.print("Run full pipeline first: [cyan]jama-scrape[/]")
            raise SystemExit(1) from None

    console.print("Pipeline stages:")
    console.print("  1. Scrape articles + glossary")
    console.print("  2. Extract entities/relationships")
    console.print("  3. Chunk for RAG retrieval")
    console.print("  4. Generate embeddings")
    console.print("  5. Export Neo4j files")
    if args.load_neo4j:
        console.print("  6. Load into Neo4j database")
    console.print()

    try:
        asyncio.run(
            run_scraper(
                output_dir=args.output,
                resume_enrichment=args.resume,
                estimate_cost=args.estimate_cost,
                use_browser=args.browser,
                load_neo4j=args.load_neo4j,
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
    except Neo4jConfigError:
        console.print("\n[red]Error: Neo4j configuration missing[/]")
        console.print("Set: [cyan]NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD[/]")
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
