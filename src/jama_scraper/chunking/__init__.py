"""LangChain-based chunking for the Jama Guide ETL pipeline.

This package provides hierarchical HTML chunking using LangChain's
HTMLHeaderTextSplitter with fallback to RecursiveCharacterTextSplitter
for large sections.
"""

from jama_scraper.chunking.adapter import create_text_splitter_adapter
from jama_scraper.chunking.config import HierarchicalChunkingConfig
from jama_scraper.chunking.hierarchical_chunker import HierarchicalHTMLSplitter

__all__ = [
    "HierarchicalChunkingConfig",
    "HierarchicalHTMLSplitter",
    "create_text_splitter_adapter",
]
