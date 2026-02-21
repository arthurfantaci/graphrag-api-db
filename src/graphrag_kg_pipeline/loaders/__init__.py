"""Custom data loaders for the guide ETL pipeline.

This package provides:
- GuideHTMLLoader: DataLoader for processing HTML/markdown content
- Article index building utilities
"""

from graphrag_kg_pipeline.loaders.html_loader import GuideHTMLLoader
from graphrag_kg_pipeline.loaders.index_builder import (
    ArticleIndex,
    build_article_index,
)

__all__ = [
    "GuideHTMLLoader",
    "ArticleIndex",
    "build_article_index",
]
