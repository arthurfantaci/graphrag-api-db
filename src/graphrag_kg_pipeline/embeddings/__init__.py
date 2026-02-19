"""Embedding providers for the knowledge graph pipeline.

This package provides:
- VoyageAIEmbeddings: Voyage AI embeddings with asymmetric input types
- create_embedder: Factory that auto-detects provider from environment
"""

from graphrag_kg_pipeline.embeddings.voyage import VoyageAIEmbeddings

__all__ = [
    "VoyageAIEmbeddings",
]
