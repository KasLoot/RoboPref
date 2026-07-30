"""OpenAI-compatible EmbeddingGemma client used by memory retrieval.

EmbeddingGemma is asymmetric for information retrieval: queries and stored
documents require different prompt prefixes.  Keeping that formatting here
prevents callers from accidentally placing query and document text in the
wrong embedding space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

from openai import OpenAI


QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: {title} | text: "
NATIVE_DIMENSIONS = 768
SUPPORTED_DIMENSIONS = frozenset({128, 256, 512, NATIVE_DIMENSIONS})


class EmbeddingServiceError(RuntimeError):
    """Base error for an unavailable or invalid embedding service."""


class EmbeddingModelDiscoveryError(EmbeddingServiceError):
    """Raised when a single usable served model cannot be resolved."""


class EmbeddingProtocolError(EmbeddingServiceError):
    """Raised when the server returns a malformed embedding response."""


@dataclass(frozen=True, slots=True)
class EmbeddingDocument:
    """One text record to index in the persistent-memory store."""

    text: str
    title: str | None = None


class EmbeddingGemmaClient:
    """Small synchronous client for a vLLM OpenAI-compatible embed server.

    Model discovery is lazy and cached.  If ``model`` is supplied, discovery
    verifies that the exact ID is served.  With no configured model, the
    server must expose exactly one model ID.

    The native 768-dimensional output is requested without an explicit
    ``dimensions`` parameter.  This works with a default EmbeddingGemma
    server and avoids requiring a Matryoshka override merely to request the
    model's full output.  Smaller dimensions are sent explicitly.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str | None = None,
        api_key: str = "EMPTY",
        dimensions: int | None = NATIVE_DIMENSIONS,
        timeout: float = 30.0,
        max_batch_size: int = 64,
        client: Any | None = None,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url:
            raise ValueError("Embedding base URL must not be empty.")
        if dimensions is not None and dimensions not in SUPPORTED_DIMENSIONS:
            allowed = ", ".join(str(value) for value in sorted(SUPPORTED_DIMENSIONS))
            raise ValueError(
                f"Embedding dimensions must be one of {allowed}, or None."
            )
        if timeout <= 0:
            raise ValueError("Embedding timeout must be positive.")
        if max_batch_size <= 0:
            raise ValueError("Embedding max_batch_size must be positive.")

        self.base_url = normalized_base_url
        self.configured_model = model.strip() if model and model.strip() else None
        self.dimensions = dimensions
        self.max_batch_size = max_batch_size
        self._resolved_model: str | None = None
        self._observed_dimensions: int | None = None
        self._client = client or OpenAI(
            base_url=self.base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
        )

    @property
    def resolved_model(self) -> str | None:
        """Return the cached model ID, or ``None`` before discovery."""

        return self._resolved_model

    @property
    def output_dimensions(self) -> int | None:
        """Return configured dimensions or the size observed from the server."""

        return self.dimensions or self._observed_dimensions

    def discover_model(self, *, force: bool = False) -> str:
        """Resolve and cache the model ID exposed by ``GET /v1/models``."""

        if self._resolved_model is not None and not force:
            return self._resolved_model

        try:
            response = self._client.models.list()
        except Exception as exc:  # SDK exceptions vary by transport/status.
            raise EmbeddingModelDiscoveryError(
                f"Could not list embedding models at {self.base_url}: {exc}"
            ) from exc

        model_ids = tuple(
            model_id
            for item in getattr(response, "data", ())
            if isinstance((model_id := getattr(item, "id", None)), str)
            and model_id
        )
        unique_model_ids = tuple(dict.fromkeys(model_ids))

        if self.configured_model is not None:
            if self.configured_model not in unique_model_ids:
                available = ", ".join(unique_model_ids) or "<none>"
                raise EmbeddingModelDiscoveryError(
                    f"Configured embedding model {self.configured_model!r} is not "
                    f"served at {self.base_url}; available models: {available}."
                )
            resolved = self.configured_model
        elif len(unique_model_ids) == 1:
            resolved = unique_model_ids[0]
        elif not unique_model_ids:
            raise EmbeddingModelDiscoveryError(
                f"No embedding model is served at {self.base_url}."
            )
        else:
            available = ", ".join(unique_model_ids)
            raise EmbeddingModelDiscoveryError(
                "Embedding model is not configured and the server exposes "
                f"multiple model IDs: {available}."
            )

        self._resolved_model = resolved
        return resolved

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query using EmbeddingGemma's query prompt."""

        content = _require_text(text, field="query")
        return self._embed_formatted([f"{QUERY_PREFIX}{content}"])[0]

    def embed_document(
        self,
        text: str,
        *,
        title: str | None = None,
    ) -> list[float]:
        """Embed one stored memory using EmbeddingGemma's document prompt."""

        return self.embed_documents([EmbeddingDocument(text=text, title=title)])[0]

    def embed_documents(
        self,
        documents: Sequence[EmbeddingDocument],
    ) -> list[list[float]]:
        """Embed stored memories in bounded batches while preserving order."""

        if not documents:
            return []
        formatted: list[str] = []
        for index, document in enumerate(documents):
            if not isinstance(document, EmbeddingDocument):
                raise TypeError(
                    "documents must contain EmbeddingDocument values; "
                    f"item {index} is {type(document).__name__}."
                )
            text = _require_text(document.text, field=f"documents[{index}].text")
            title = _format_title(document.title)
            formatted.append(f"{DOCUMENT_PREFIX.format(title=title)}{text}")
        return self._embed_formatted(formatted)

    def _embed_formatted(self, inputs: Sequence[str]) -> list[list[float]]:
        model = self.discover_model()
        embeddings: list[list[float]] = []
        for offset in range(0, len(inputs), self.max_batch_size):
            batch = list(inputs[offset : offset + self.max_batch_size])
            request: dict[str, Any] = {
                "model": model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.dimensions is not None and self.dimensions != NATIVE_DIMENSIONS:
                request["dimensions"] = self.dimensions

            try:
                response = self._client.embeddings.create(**request)
            except Exception as exc:  # SDK exceptions vary by transport/status.
                raise EmbeddingServiceError(
                    f"Embedding request to {self.base_url} failed: {exc}"
                ) from exc
            embeddings.extend(self._validate_response(response, expected=len(batch)))
        return embeddings

    def _validate_response(
        self,
        response: Any,
        *,
        expected: int,
    ) -> list[list[float]]:
        raw_items = list(getattr(response, "data", ()))
        if len(raw_items) != expected:
            raise EmbeddingProtocolError(
                "Embedding server returned "
                f"{len(raw_items)} vectors for {expected} inputs."
            )

        indexed_items: dict[int, Any] = {}
        for item in raw_items:
            index = getattr(item, "index", None)
            if not isinstance(index, int) or isinstance(index, bool):
                raise EmbeddingProtocolError(
                    "Embedding response contains a non-integer index."
                )
            if index in indexed_items:
                raise EmbeddingProtocolError(
                    f"Embedding response repeats index {index}."
                )
            indexed_items[index] = item

        expected_indices = set(range(expected))
        if set(indexed_items) != expected_indices:
            raise EmbeddingProtocolError(
                "Embedding response indices do not match the input batch."
            )

        vectors: list[list[float]] = []
        for index in range(expected):
            raw_vector = getattr(indexed_items[index], "embedding", None)
            if not isinstance(raw_vector, (list, tuple)) or not raw_vector:
                raise EmbeddingProtocolError(
                    f"Embedding at index {index} is not a non-empty float vector."
                )
            try:
                vector = [float(value) for value in raw_vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingProtocolError(
                    f"Embedding at index {index} contains a non-numeric value."
                ) from exc
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingProtocolError(
                    f"Embedding at index {index} contains a non-finite value."
                )
            if not any(value != 0.0 for value in vector):
                raise EmbeddingProtocolError(
                    f"Embedding at index {index} is the zero vector."
                )

            dimension = len(vector)
            expected_dimension = self.dimensions or self._observed_dimensions
            if expected_dimension is not None and dimension != expected_dimension:
                raise EmbeddingProtocolError(
                    f"Embedding at index {index} has {dimension} dimensions; "
                    f"expected {expected_dimension}."
                )
            if self._observed_dimensions is None:
                self._observed_dimensions = dimension
            elif dimension != self._observed_dimensions:
                raise EmbeddingProtocolError(
                    "Embedding dimensions changed between server responses: "
                    f"{self._observed_dimensions} then {dimension}."
                )
            vectors.append(vector)
        return vectors


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity with strict shape and finite-value checks."""

    if not left or not right:
        raise ValueError("Cosine similarity requires two non-empty vectors.")
    if len(left) != len(right):
        raise ValueError(
            "Cosine similarity requires vectors of equal length; "
            f"received {len(left)} and {len(right)}."
        )
    try:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cosine similarity vectors must be numeric.") from exc
    if not all(
        math.isfinite(value) for value in (*left_values, *right_values)
    ):
        raise ValueError("Cosine similarity vectors must contain finite values.")

    dot = math.fsum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    result = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, result))


def _require_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    if not value.strip():
        raise ValueError(f"{field} must not be empty.")
    return value


def _format_title(value: str | None) -> str:
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise TypeError("document title must be a string or None.")
    return value.strip() or "none"
