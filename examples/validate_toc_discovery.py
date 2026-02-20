#!/usr/bin/env python3
"""Validate dynamic TOC discovery against the live Jama guide website.

Fetches the guide's main page, parses ``div#chapter-menu``, and prints
the discovered chapter/article structure. Use this as a quick sanity
check before re-ingestion to confirm the guide's HTML structure hasn't
changed in a way that breaks the parser.

Usage:
    uv run python examples/validate_toc_discovery.py

Requires:
    - Network access to jamasoftware.com (no API keys needed)

Exit codes:
    0 — TOC parsed successfully
    1 — Fetch or parse failed
"""

from __future__ import annotations

import sys

import httpx
from rich.console import Console
from rich.table import Table

from graphrag_kg_pipeline.config import BASE_URL
from graphrag_kg_pipeline.exceptions import ScraperError
from graphrag_kg_pipeline.parser import HTMLParser

MAX_URL_DISPLAY_LENGTH = 80

console = Console()


def main() -> int:
    """Fetch the guide page and validate TOC discovery."""
    console.print(f"\n[bold]Fetching guide page:[/] {BASE_URL}")

    headers = {"User-Agent": "JamaGuideScraper/0.1.0 (Educational/Research)"}
    try:
        response = httpx.get(BASE_URL, headers=headers, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to fetch guide page:[/] {e}")
        return 1

    console.print(f"  Status: [green]{response.status_code}[/]  ({len(response.text):,} bytes)\n")

    parser = HTMLParser()
    try:
        chapters = parser.parse_chapter_menu(response.text)
    except ScraperError as e:
        console.print(f"[red]TOC parse failed:[/] {e}")
        return 1

    # Summary table
    table = Table(title="Discovered Guide Structure", show_lines=True)
    table.add_column("Ch", justify="right", style="bold")
    table.add_column("Chapter Title")
    table.add_column("Slug")
    table.add_column("Articles", justify="right")

    total_articles = 0
    for ch in chapters:
        table.add_row(
            str(ch.number),
            ch.title,
            ch.slug,
            str(len(ch.articles)),
        )
        total_articles += len(ch.articles)

    console.print(table)
    console.print(
        f"\n[bold green]Discovered {len(chapters)} chapters, "
        f"{total_articles} articles (including overviews)[/]\n"
    )

    # Detail: show first 3 chapters' articles as a spot check
    for ch in chapters[:3]:
        console.print(f"[bold]Chapter {ch.number}: {ch.title}[/]")
        for art in ch.articles:
            number_str = f"  {art.number}." if art.number > 0 else "  ov"
            url_snippet = (
                art.url[:MAX_URL_DISPLAY_LENGTH] + "..."
                if art.url and len(art.url) > MAX_URL_DISPLAY_LENGTH
                else art.url
            )
            console.print(f"  {number_str} {art.title}")
            console.print(f"       [dim]{url_snippet}[/]")
        console.print()

    # Cross-chapter URL check: flag any article whose URL doesn't contain
    # its parent chapter slug (these are valid but worth highlighting)
    cross_chapter = []
    for ch in chapters:
        for art in ch.articles:
            if art.url and f"/{ch.slug}/" not in art.url:
                cross_chapter.append((ch, art))

    if cross_chapter:
        console.print("[yellow]Cross-chapter URLs detected (expected for some articles):[/]")
        for ch, art in cross_chapter:
            console.print(f"  Ch{ch.number} art{art.number}: {art.url}")
        console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
