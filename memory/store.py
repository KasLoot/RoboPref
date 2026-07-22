"""Compatibility exports for the redesigned repositories."""

from memory.repositories import (
    HistoryOutboxRepository,
    HistoryRepository,
    MemoryRepositoryError,
    PreferenceRepository,
)

__all__ = [
    "HistoryRepository",
    "HistoryOutboxRepository",
    "MemoryRepositoryError",
    "PreferenceRepository",
]
