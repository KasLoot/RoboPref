"""Persistent-memory infrastructure for PrefMem."""

from prefmem.memory.embeddings import (
    EmbeddingDocument,
    EmbeddingGemmaClient,
    EmbeddingModelDiscoveryError,
    EmbeddingProtocolError,
    EmbeddingServiceError,
    cosine_similarity,
)
from prefmem.memory.service import (
    MemoryAuthorizationError,
    PrefMemMemoryService,
)
from prefmem.memory.store import MarkdownMemoryStore, MemoryFormatError

__all__ = [
    "EmbeddingDocument",
    "EmbeddingGemmaClient",
    "EmbeddingModelDiscoveryError",
    "EmbeddingProtocolError",
    "EmbeddingServiceError",
    "MarkdownMemoryStore",
    "MemoryAuthorizationError",
    "MemoryFormatError",
    "PrefMemMemoryService",
    "cosine_similarity",
]
