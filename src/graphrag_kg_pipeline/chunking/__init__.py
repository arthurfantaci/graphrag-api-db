"""LangChain-based chunking for the guide ETL pipeline.

This package provides hierarchical HTML chunking using LangChain's
HTMLHeaderTextSplitter with fallback to RecursiveCharacterTextSplitter
for large sections.
"""

from graphrag_kg_pipeline.chunking.adapter import create_text_splitter_adapter
from graphrag_kg_pipeline.chunking.config import HierarchicalChunkingConfig
from graphrag_kg_pipeline.chunking.hierarchical_chunker import HierarchicalHTMLSplitter

__all__ = [
    "HierarchicalChunkingConfig",
    "HierarchicalHTMLSplitter",
    "create_text_splitter_adapter",
]
