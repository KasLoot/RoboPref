import copy
import uuid

from collections.abc import Callable, Mapping

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage
from typing_extensions import TypedDict, Annotated, Literal
from prefmem.agents.config import Memory_Config
from langchain.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
import operator
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from IPython.display import Image, display
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.runtime import Runtime
from prefmem.agents.vision import image_data_url, get_start_end_frames, get_live_frame
import json

import numpy as np

from time import perf_counter

from prefmem.agents.metrics import TurnMetrics


from colorama import init
from termcolor import colored
init()



class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class CurrentFrameContext(TypedDict):
    current_frame: dict


def _parse_memory_json_object(response_content: object) -> dict:
    """Decode one Memory JSON object with an optional transport fence.

    Gemma occasionally wraps an otherwise valid JSON-only response in a
    single Markdown fence.  Accept that one wrapper, but reject prose outside
    it, nested/additional fences, non-object JSON, and malformed content.  The
    caller still performs the full candidate provenance and schema checks.
    """

    if not isinstance(response_content, str):
        raise ValueError("Memory retrieval response must be a JSON string")
    text = response_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip().casefold() not in {"```", "```json"}
            or lines[-1].strip() != "```"
            or any("```" in line for line in lines[1:-1])
        ):
            raise ValueError(
                "Memory retrieval response must contain one JSON object"
            )
        text = "\n".join(lines[1:-1]).strip()
    elif "```" in text:
        raise ValueError("Memory retrieval response contains an invalid fence")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Memory retrieval response is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Memory retrieval response must be a JSON object")
    return dict(payload)

class VLLMChatOpenAI(ChatOpenAI):
    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        choices = (
            chunk.get("choices")
            or chunk.get("chunk", {}).get("choices")
            or []
        )
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            reasoning = (
                delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if reasoning:
                generation.message.additional_kwargs["reasoning"] = reasoning

        request_metrics = (
            chunk.get("metrics")
            or chunk.get("chunk", {}).get("metrics")
        )
        if generation is not None and request_metrics:
            generation.message.response_metadata[
                "vllm_metrics"
            ] = request_metrics

        return generation


class VLLMEmbeddingGemma:
    """SentenceTransformer-style adapter for a vLLM embeddings endpoint."""

    QUERY_PROMPT = "task: search result | query: "
    DOCUMENT_PROMPT = "title: none | text: "

    def __init__(
        self,
        model: str,
        base_url: str,
    ) -> None:
        self._client = OpenAIEmbeddings(
            model=model,
            api_key="EMPTY",
            base_url=base_url,
            # vLLM must tokenize with EmbeddingGemma's tokenizer. LangChain's
            # length-safe path uses tiktoken and may send incompatible token IDs.
            check_embedding_ctx_length=False,
        )

    def encode_document(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(
            [f"{self.DOCUMENT_PROMPT}{text}" for text in texts]
        )

    def encode_query(
        self,
        texts: str | list[str],
    ) -> list[float] | list[list[float]]:
        single_input = isinstance(texts, str)
        batch = [texts] if single_input else texts
        embeddings = self._client.embed_documents(
            [f"{self.QUERY_PROMPT}{text}" for text in batch]
        )
        return embeddings[0] if single_input else embeddings


class Memory_Agent:
    def __init__(
        self,
        model_config: str = "vllm",
        args=None,
        metrics: TurnMetrics | None = None,
    ):
        self.config = Memory_Config(
            model_config,
            model=getattr(args, "model", None),
            model_base_url=getattr(args, "model_base_url", None),
        )
        self.args = args
        self.metrics = metrics
        self.llm = VLLMChatOpenAI(
            model=self.config.model,
            api_key="EMPTY",
            base_url=self.config.model_base_url,
            max_tokens=2048,  # Smaller while debugging
            temperature=0,
            streaming=True,
            stream_usage=True,
            timeout=300,
        )


        self.remember_tool = StructuredTool.from_function(func=self.remember)
        self.forget_tool = StructuredTool.from_function(func=self.forget)
        self.update_tool = StructuredTool.from_function(func=self.update)
        self.retrieve_tool = StructuredTool.from_function(func=self.retrieve)
        self.TOOLS = [self.remember_tool, self.forget_tool, self.update_tool, self.retrieve_tool]
        self.TOOLS_BY_NAME = {tool.name: tool for tool in self.TOOLS}
        self.llm = self.llm.bind_tools(self.TOOLS)
        self.agent = self.build_agent()
        self.system_prompt = self.config.system_prompt
        self.thinking_enabled = bool(args.think and (args.think == "all" or "Memory" in args.think))
        self.print_raw = bool(getattr(args, "print_raw", False))
        self._trace_sink: Callable[[dict], None] | None = None
        self.last_retrieval_trace: dict | None = None

        self.initialize()


    def initialize(self):
        memory_store_path = self.args.memory_store_path

        if memory_store_path:
            self.pref_json_path = Path(memory_store_path) / "preference.json"
            self.pref_embedding_path = Path(memory_store_path) / "preference.npy"

        store_is_complete = bool(
            memory_store_path
            and self.pref_json_path.is_file()
            and self.pref_embedding_path.is_file()
        )
        store_is_partial = bool(
            memory_store_path
            and self.pref_json_path.exists() != self.pref_embedding_path.exists()
        )
        if store_is_partial:
            raise ValueError(
                "The preference store is incomplete: preference.json and "
                "preference.npy must either both exist or both be absent"
            )

        if store_is_complete:
            with open(
                self.pref_json_path,
                "r",
                encoding="utf-8",
            ) as file:
                self.pref_json = json.load(file)

            with open(
                self.pref_embedding_path,
                "rb",
            ) as file:
                self.pref_embedding = np.load(file)

            self.pref_embedding = np.asarray(
                self.pref_embedding,
                dtype=np.float32,
                order="C",
            )

            if not isinstance(self.pref_json, list):
                raise ValueError("preference.json must contain a JSON list")
            if self.pref_embedding.ndim != 2:
                raise ValueError("preference.npy must contain a 2D array")
            if self.pref_embedding.shape[0] != len(self.pref_json):
                raise ValueError(
                    "Stored preference and embedding counts do not match"
                )
            if self.pref_embedding.shape[1] != self.config.embedding_dimensions:
                raise ValueError(
                    "Stored preference embeddings have dimension "
                    f"{self.pref_embedding.shape[1]}; expected "
                    f"{self.config.embedding_dimensions}"
                )

            norms = np.linalg.norm(
                self.pref_embedding,
                axis=1,
                keepdims=True,
            )

            if np.any(norms == 0):
                zero_indices = np.flatnonzero(norms[:, 0] == 0)
                raise ValueError(
                    "Stored zero-length embeddings at indices "
                    f"{zero_indices.tolist()}"
                )

            self.pref_embedding /= norms

        else:
            self.pref_json = []

            # Prefer a consistently 2D empty representation.
            self.pref_embedding = np.empty(
                (0, self.config.embedding_dimensions),
                dtype=np.float32,
            )

        embedding_model = getattr(
            self.args,
            "embedding_model",
            self.config.embedding_model,
        ) or self.config.embedding_model
        embedding_model_base_url = getattr(
            self.args,
            "embedding_model_base_url",
            self.config.embedding_model_base_url,
        ) or self.config.embedding_model_base_url
        self.config.embedding_model = embedding_model
        self.config.embedding_model_base_url = embedding_model_base_url
        self.embedding_model = VLLMEmbeddingGemma(
            model=embedding_model,
            base_url=embedding_model_base_url,
        )

        self.top_k = self.config.top_k
        self.top_cap_k = self.config.top_cap_k
        self._reset_request_state()

    def _reset_request_state(self, request_type: str | None = None) -> None:
        """Clear retrieval evidence before handling a new memory request."""
        self._request_type = request_type
        self._request_id = uuid.uuid4().hex
        self._original_request: str | None = None
        self._retrieval_performed = False
        self._retrieved_memories: dict[str, dict] = {}
        self.last_retrieval_trace = None

    def set_trace_sink(
        self,
        sink: Callable[[dict], None] | None,
    ) -> None:
        """Install an experiment observer without changing Memory responses."""

        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable or None")
        self._trace_sink = sink

    def _publish_trace(self, trace: dict) -> None:
        frozen = copy.deepcopy(trace)
        self.last_retrieval_trace = frozen
        sink = getattr(self, "_trace_sink", None)
        if sink is not None:
            sink(copy.deepcopy(frozen))

    def _validate_mutation(self, memory_id: str | None = None) -> dict | None:
        if self._request_type != "MUTATE":
            return {
                "status": "INVALID_REQUEST",
                "message": "Write tools require a MUTATE REQUEST.",
            }

        if not self._retrieval_performed:
            return {
                "status": "INVALID_REQUEST",
                "message": "A mutation request must retrieve memories first.",
            }

        if memory_id is not None and memory_id not in self._retrieved_memories:
            return {
                "status": "INVALID_REQUEST",
                "message": (
                    "The mutation target ID was not returned by this request's "
                    "retrieval."
                ),
            }

        return None

    def write_memory_to_disk(self):
        if self.args.memory_store_path:
            self.pref_json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(
                self.pref_json_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(self.pref_json, file, ensure_ascii=False, indent=4)

            np.save(
                self.pref_embedding_path,
                self.pref_embedding,
                allow_pickle=False,
            )

    def remember(self, text: str) -> dict:
        """
        Store a new memory after the current request has retrieved candidates.

        Parameters
        ----------
        text: str
            The content of the memory to store.

        Returns
        -------
        dict
            A structured remembered, unchanged, or invalid-request result.
        """
        validation_error = self._validate_mutation()
        if validation_error is not None:
            return validation_error

        if not isinstance(text, str) or not text.strip():
            return {
                "status": "INVALID_REQUEST",
                "message": "text must be a non-empty string.",
            }

        normalized_text = text.strip().casefold()
        for existing_memory in self.pref_json:
            if (
                existing_memory.get("text", "").strip().casefold()
                == normalized_text
            ):
                return {
                    "status": "UNCHANGED",
                    "id": existing_memory["id"],
                    "text": existing_memory["text"],
                    "message": "An identical memory already exists.",
                }

        memory = {
            "id": f"pref-{uuid.uuid4()}",
            "text": text,
        }

        embedding = self.embedding_model.encode_document([text])[0]
        embedding = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("The embedding model produced a zero vector")

        embedding /= norm

        self.pref_json.append(memory)

        if self.pref_embedding.shape[0] == 0:
            self.pref_embedding = embedding.reshape(1, -1)
        else:
            self.pref_embedding = np.vstack(
                (self.pref_embedding, embedding)
            )

        self.write_memory_to_disk()

        return {
            "status": "REMEMBERED",
            "id": memory["id"],
            "text": memory["text"],
        }

    
    def forget(self, memory_id: str) -> dict:
        """
        Remove a memory selected by the current request's retrieval.

        Parameters
        ----------
        memory_id: str
            An ID returned by the current request's retrieval.

        Returns
        -------
        dict
            A structured forgotten or invalid-request result.
        """
        validation_error = self._validate_mutation(memory_id)
        if validation_error is not None:
            return validation_error

        retrieved_memory = self._retrieved_memories[memory_id]
        removed_memory = self._forget_by_id(memory_id)

        return {
            "status": "FORGOTTEN",
            "id": removed_memory["id"],
            "forgotten_text": removed_memory["text"],
            "similarity": retrieved_memory["similarity"],
        }

    def _forget_by_id(self, memory_id: str) -> dict:
        index_to_remove = None

        for index, memory in enumerate(self.pref_json):
            if memory.get("id") == memory_id:
                index_to_remove = index
                break

        if index_to_remove is None:
            raise ValueError(f"No memory found with ID: {memory_id}")

        removed_memory = self.pref_json.pop(index_to_remove)
        self.pref_embedding = np.delete(
            self.pref_embedding, index_to_remove, axis=0
        )
        self.write_memory_to_disk()

        return removed_memory

    
    def update(self, memory_id: str, new_text: str) -> dict:
        """
        Update a memory selected by the current request's retrieval.

        Parameters
        ----------
        memory_id: str
            An ID returned by the current request's retrieval.
        new_text: str
            The complete replacement text for the memory.

        Returns
        -------
        dict
            A structured updated, unchanged, or invalid-request result.
        """
        validation_error = self._validate_mutation(memory_id)
        if validation_error is not None:
            return validation_error

        if not isinstance(new_text, str) or not new_text.strip():
            return {
                "status": "INVALID_REQUEST",
                "message": "new_text must be a non-empty string.",
            }

        retrieved_memory = self._retrieved_memories[memory_id]
        if (
            retrieved_memory["text"].strip().casefold()
            == new_text.strip().casefold()
        ):
            return {
                "status": "UNCHANGED",
                "id": memory_id,
                "text": retrieved_memory["text"],
                "message": "The selected memory already has this text.",
            }

        previous_text = self._update_by_id(
            memory_id,
            new_text,
        )

        return {
            "status": "UPDATED",
            "id": memory_id,
            "previous_text": previous_text,
            "new_text": new_text,
            "similarity": retrieved_memory["similarity"],
        }

    def _update_by_id(self, memory_id: str, new_text: str) -> str:
        index_to_update = None

        for index, memory in enumerate(self.pref_json):
            if memory.get("id") == memory_id:
                index_to_update = index
                break

        if index_to_update is None:
            raise ValueError(f"No memory found with ID: {memory_id}")

        new_embedding = self.embedding_model.encode_document([new_text])[0]
        new_embedding = np.asarray(new_embedding, dtype=np.float32)
        norm = np.linalg.norm(new_embedding)

        if norm == 0:
            raise ValueError("The embedding model produced a zero vector")

        new_embedding /= norm

        previous_text = self.pref_json[index_to_update]["text"]
        self.pref_json[index_to_update]["text"] = new_text
        self.pref_embedding[index_to_update] = new_embedding
        self.write_memory_to_disk()

        return previous_text

    def retrieve(self, query: list[str]) -> list[dict]:
        """
        Retrieve memories using multiple query phrasings.

        Each query independently retrieves self.top_k candidates. Candidates from
        all queries are merged, deduplicated by memory ID, ranked using their
        highest cosine similarity, and capped at self.top_cap_k final results.

        Parameters
        ----------
        query:
            Different phrasings of the same memory request.

        Returns
        -------
        list[dict]
            Deduplicated memories ordered by descending similarity:

            [
                {
                    "id": "pref-...",
                    "memory_type": "PREFERENCE",
                    "text": "...",
                    "similarity": 0.81,
                },
                ...
            ]
        """
        if self._request_type not in {"RETRIEVE", "MUTATE"}:
            raise ValueError(
                "Memory requests must start with RETRIEVE REQUEST: or "
                "MUTATE REQUEST:."
            )
        if self._retrieval_performed:
            raise ValueError("retrieve may only be called once per memory request")
        if not query:
            raise ValueError("query must contain at least one query string")

        results = self._retrieve(query)
        self._retrieval_performed = True
        self._retrieved_memories = {
            result["id"]: result
            for result in results
            if result.get("id") is not None
        }
        return results

    def _retrieve(
        self,
        query: list[str],
    ) -> list[dict]:
        queries = list(query)
        batch_size = len(queries)
        if batch_size == 0:
            return []
        if any(not isinstance(item, str) or not item.strip() for item in queries):
            raise ValueError("every retrieval query must be a non-empty string")
        queries = [item.strip() for item in queries]

        number_of_documents = len(self.pref_json)
        trace: dict = {
            "schema_version": 1,
            "stage": "RETRIEVAL_CAPPED",
            "request_id": getattr(self, "_request_id", None) or uuid.uuid4().hex,
            "request_type": getattr(self, "_request_type", None),
            "original_request": getattr(self, "_original_request", None),
            "query_count": batch_size,
            "queries": queries,
            "n_current": number_of_documents,
            "top_k_configured": int(self.top_k),
            "top_k_effective": min(int(self.top_k), number_of_documents),
            "top_cap_k": int(self.top_cap_k),
            "per_query_candidates": [],
            "merged_before_cap": [],
            "capped_candidates": [],
            "semantic_decisions": [],
            "final_memory_ids": [],
            "timing_seconds": {
                "embedding": 0.0,
                "similarity_search": 0.0,
                "merge_deduplicate_cap": 0.0,
                "total_retrieval": 0.0,
            },
        }

        if self.pref_embedding.size == 0 or not self.pref_json:
            self._publish_trace(trace)
            return []

        document_embeddings = np.asarray(
            self.pref_embedding,
            dtype=np.float32,
            order="C",
        )

        if document_embeddings.ndim != 2:
            raise ValueError(
                "self.pref_embedding must have shape "
                "(number_of_documents, embedding_dimension)"
            )

        embedding_count, embedding_dimension = document_embeddings.shape

        if number_of_documents != embedding_count:
            raise ValueError(
                f"Embedding count ({embedding_count}) does not match "
                f"memory count ({len(self.pref_json)})"
            )

        if self.top_k <= 0:
            raise ValueError("self.top_k must be positive")

        if self.top_cap_k <= 0:
            raise ValueError("self.top_cap_k must be positive")

        # A query cannot retrieve more documents than currently exist.
        per_query_top_k = min(self.top_k, number_of_documents)

        embedding_start = perf_counter()

        query_embeddings = self.embedding_model.encode_query(queries)
        query_embeddings = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        embedding_elapsed = perf_counter() - embedding_start

        # Some embedding implementations return shape (D,) for one query.
        if query_embeddings.ndim == 1:
            if batch_size != 1:
                raise ValueError(
                    "Embedding model returned one vector for multiple queries"
                )

            query_embeddings = query_embeddings.reshape(1, -1)

        expected_shape = (batch_size, embedding_dimension)

        if query_embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected query embeddings with shape {expected_shape}, "
                f"received {query_embeddings.shape}"
            )

        # Normalize queries. Stored document embeddings must also be normalized.
        query_norms = np.linalg.norm(
            query_embeddings,
            axis=1,
            keepdims=True,
        )

        zero_query_indices = np.flatnonzero(query_norms[:, 0] <= 0)

        if zero_query_indices.size:
            raise ValueError(
                "Zero-length query embeddings at indices "
                f"{zero_query_indices.tolist()}"
            )

        query_embeddings /= query_norms

        query_embeddings = np.ascontiguousarray(
            query_embeddings,
            dtype=np.float32,
        )

        search_start = perf_counter()

        # Shapes:
        #
        # document_embeddings: (N, D)
        # query_embeddings.T:  (D, B)
        # similarities:        (N, B)
        #
        # similarities[n, b] is the cosine similarity between
        # memory n and query phrasing b.
        similarities = document_embeddings @ query_embeddings.T

        # Retrieve self.top_k candidates independently for every query.
        partition_start = number_of_documents - per_query_top_k

        unordered_top_indices = np.argpartition(
            similarities,
            kth=partition_start,
            axis=0,
        )[partition_start:, :]

        search_elapsed = perf_counter() - search_start

        merge_start = perf_counter()

        # memory_id -> best candidate for that memory.
        #
        # Using max similarity is appropriate because the query strings are
        # alternative phrasings of the same request. A memory only needs to match
        # one phrasing strongly to be retained.
        deduplicated: dict[str, dict] = {}

        for query_index in range(batch_size):
            ranked_document_indices = sorted(
                (
                    int(document_index)
                    for document_index in unordered_top_indices[:, query_index]
                ),
                key=lambda document_index: (
                    -float(similarities[document_index, query_index]),
                    document_index,
                ),
            )
            for rank, document_index in enumerate(
                ranked_document_indices,
                start=1,
            ):
                similarity = float(similarities[document_index, query_index])

                memory = self.pref_json[document_index]
                memory_id = memory.get("id")
                trace["per_query_candidates"].append(
                    {
                        "query_index": query_index,
                        "query": queries[query_index],
                        "memory_id": memory_id,
                        "document_index": document_index,
                        "rank": rank,
                        "similarity": similarity,
                    }
                )

                # Do not accidentally collapse memories that have no ID.
                if memory_id is None:
                    deduplication_key = f"document-index:{document_index}"
                else:
                    deduplication_key = str(memory_id)

                existing = deduplicated.get(deduplication_key)

                candidate = {
                    "id": memory_id,
                    "memory_type": "PREFERENCE",
                    "text": memory["text"],
                    "similarity": similarity,
                    "best_query_index": query_index,
                    "best_query_rank": rank,
                    "document_index": document_index,
                }
                if existing is None or (
                    similarity,
                    -query_index,
                    -rank,
                    -document_index,
                ) > (
                    existing["similarity"],
                    -existing["best_query_index"],
                    -existing["best_query_rank"],
                    -existing["document_index"],
                ):
                    deduplicated[deduplication_key] = {
                        **candidate,
                    }

        # Rank all unique candidates globally.
        merged_before_cap = sorted(
            deduplicated.values(),
            key=lambda item: (
                -item["similarity"],
                "" if item["id"] is None else str(item["id"]),
                item["document_index"],
            ),
        )
        capped_candidates = merged_before_cap[: self.top_cap_k]
        final_results = [
            {
                "id": item["id"],
                "memory_type": item["memory_type"],
                "text": item["text"],
                "similarity": item["similarity"],
            }
            for item in capped_candidates
        ]

        merge_elapsed = perf_counter() - merge_start
        total_elapsed = (
            embedding_elapsed
            + search_elapsed
            + merge_elapsed
        )
        trace["merged_before_cap"] = copy.deepcopy(merged_before_cap)
        trace["capped_candidates"] = copy.deepcopy(final_results)
        trace["timing_seconds"] = {
            "embedding": embedding_elapsed,
            "similarity_search": search_elapsed,
            "merge_deduplicate_cap": merge_elapsed,
            "total_retrieval": total_elapsed,
        }
        self._publish_trace(trace)

        print(
            f"Memory retrieval: "
            f"{batch_size} query phrasings, "
            f"{number_of_documents:,} memories, "
            f"top-{per_query_top_k} per query, "
            f"{len(deduplicated)} unique candidates, "
            f"{len(final_results)} final results"
        )
        print(
            f"Embedding: {embedding_elapsed:.6f}s | "
            f"Similarity search: {search_elapsed:.6f}s | "
            f"Merge/deduplicate: {merge_elapsed:.6f}s | "
            f"Total: {total_elapsed:.6f}s"
        )

        return final_results

    def _finalize_semantic_trace(self, response_content: object) -> None:
        """Validate that the Memory model only returns capped candidates."""

        if getattr(self, "_request_type", None) != "RETRIEVE":
            return
        trace = getattr(self, "last_retrieval_trace", None)
        if trace is None:
            raise ValueError("Memory retrieval completed without a retrieval trace")
        payload = _parse_memory_json_object(response_content)

        raw_returned = payload.get("retrieved_memory")
        if not isinstance(raw_returned, list):
            raise ValueError("Memory retrieval response must contain retrieved_memory")

        capped = trace.get("capped_candidates", [])
        capped_by_id = {
            item.get("id"): item
            for item in capped
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        returned_ids: list[str] = []
        for index, raw in enumerate(raw_returned):
            if not isinstance(raw, Mapping):
                raise ValueError(f"retrieved_memory[{index}] must be an object")
            memory_id = raw.get("id")
            if not isinstance(memory_id, str) or memory_id not in capped_by_id:
                raise ValueError(
                    f"retrieved_memory[{index}] is not from the capped candidates"
                )
            if memory_id in returned_ids:
                raise ValueError("Memory retrieval response contains duplicate IDs")
            expected = capped_by_id[memory_id]
            for field in ("memory_type", "text"):
                if raw.get(field) != expected.get(field):
                    raise ValueError(
                        f"retrieved_memory[{index}].{field} changed capped data"
                    )
            try:
                similarity = float(raw.get("similarity"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"retrieved_memory[{index}].similarity must be numeric"
                ) from error
            if not np.isclose(
                similarity,
                float(expected["similarity"]),
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError(
                    f"retrieved_memory[{index}].similarity changed capped data"
                )
            returned_ids.append(memory_id)

        status = payload.get("status")
        if status == "EMPTY" and returned_ids:
            raise ValueError("EMPTY Memory response cannot return candidates")
        if status == "FOUND" and not returned_ids:
            raise ValueError("FOUND Memory response must return a candidate")
        if status not in {"FOUND", "EMPTY"}:
            raise ValueError("Memory retrieval status must be FOUND or EMPTY")

        returned_set = set(returned_ids)
        updated = copy.deepcopy(trace)
        updated["stage"] = "SEMANTIC_FILTER_COMPLETE"
        updated["semantic_decisions"] = [
            {
                "memory_id": item.get("id"),
                "label": (
                    "APPLICABLE_CONTEXT"
                    if item.get("id") in returned_set
                    else "IRRELEVANT"
                ),
                "returned": item.get("id") in returned_set,
                "label_source": "inferred_from_validated_return_set",
            }
            for item in capped
        ]
        updated["final_memory_ids"] = returned_ids
        self._publish_trace(updated)

    

    def build_agent(self):
        agent_builder = StateGraph(
            MessagesState, 
            context_schema=CurrentFrameContext
        )
        agent_builder.add_node("Memory Agent", self.llm_call)
        agent_builder.add_node("Tool Node", self.tool_node)

        agent_builder.add_edge(START, "Memory Agent")
        agent_builder.add_conditional_edges(
            "Memory Agent", 
            self.should_continue, 
            ["Tool Node", END]
            )
        agent_builder.add_edge("Tool Node", "Memory Agent")

        checkpointer = InMemorySaver()
        
        agent = agent_builder.compile(checkpointer=checkpointer)

        return agent
    
    

    def tool_node(self, state: dict):
                """Performs the tool call"""
        
                result = []
                for tool_call in state["messages"][-1].tool_calls:
                    try:
                        tool = self.TOOLS_BY_NAME[tool_call["name"]]
                        observation = tool.invoke(tool_call["args"])
                    except Exception as exc:
                        observation = {
                            "status": "ERROR",
                            "tool": tool_call["name"],
                            "message": str(exc),
                        }

                    if not isinstance(observation, str):
                        observation = json.dumps(
                            observation,
                            ensure_ascii=False,
                            default=str,
                        )
                    result.append(
                        ToolMessage(
                            content=observation,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                return {"messages": result}
            
            
    def llm_call(self, state: dict, runtime: Runtime[CurrentFrameContext]):
        """LLM decides whether to call a tool or not"""

        model_messages = list(state["messages"])
        request_options = (
            {"reasoning_effort": "high"}
            if getattr(self, "thinking_enabled", False)
            else {}
        )
        for index in range(len(model_messages) - 1, -1, -1):
            message = model_messages[index]
            if isinstance(message, HumanMessage):
                model_messages[index] = message.model_copy(
                    update={
                        "content": [
                            runtime.context["current_frame"],
                            {"type": "text", "text": message.content},
                        ]
                    }
                )
                break

        prompt_messages = [
            SystemMessage(
                content=getattr(
                    self,
                    "system_prompt",
                    self.config.system_prompt,
                )
            )
        ] + model_messages

        started = perf_counter()
        answer = self.llm.invoke(
            prompt_messages,
            **request_options,
        )
        elapsed_seconds = perf_counter() - started

        if self.metrics is not None:
            self.metrics.record(
                agent="Memory Agent",
                prompt_messages=prompt_messages,
                response=answer,
                elapsed_seconds=elapsed_seconds,
            )

        return {
            "messages": [answer],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }


    def should_continue(self, state: MessagesState):
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "Tool Node"

        # Otherwise, we stop (reply to the user)
        return END


    def invoke_agent(self, messages, current_frame) -> dict:
        # ``InMemorySaver`` requires a thread identifier for every graph
        # invocation.  Memory requests are deliberately isolated from one
        # another: retrieval/mutation authority is reset per request and prior
        # tool traces must not leak into the next semantic decision.
        request_id = getattr(self, "_request_id", None) or uuid.uuid4().hex
        config = {
            "configurable": {"thread_id": f"memory-request-{request_id}"},
            "recursion_limit": 20,
        }

        active_section = None
        active_tool_index = None
        streamed_content = False
        streamed_reasoning = False
        streamed_tool_calls = False
        raw_response_count = 0

        def start_section(label):
            nonlocal active_section
            if active_section == label:
                return
            if active_section is not None:
                print()
            print(colored(f"{label}:", "black", "on_white"), flush=True)
            active_section = label

        final_state = None
        for part in self.agent.stream(
            input={"messages": messages},
            config=config,
            context={"current_frame": current_frame},
            stream_mode=["messages", "updates", "values"],
            version="v2",
        ):
            if part["type"] == "values":
                final_state = part["data"]
                continue

            if part["type"] == "messages":
                chunk, metadata = part["data"]

                # Ignore messages belonging to any nested agent or tool.
                if not isinstance(chunk, AIMessage):
                    continue

                if metadata.get("langgraph_node") != "Memory Agent":
                    continue

                reasoning = chunk.additional_kwargs.get("reasoning")
                if reasoning:
                    start_section("Memory Agent Thinking")
                    print(reasoning, end="", flush=True)
                    streamed_reasoning = True

                if isinstance(chunk.content, str) and chunk.content:
                    start_section("Memory Agent Response")
                    print(chunk.content, end="", flush=True)
                    streamed_content = True

                for call in getattr(chunk, "tool_call_chunks", []):
                    start_section("Memory Agent Tool Call")
                    call_index = call.get("index")
                    if call_index != active_tool_index:
                        if active_tool_index is not None:
                            print()
                        print(f"[{call_index}] ", end="", flush=True)
                        active_tool_index = call_index
                    if call.get("name"):
                        print(
                            f"{call['name']} arguments=",
                            end="",
                            flush=True,
                        )
                    if call.get("args"):
                        print(call["args"], end="", flush=True)
                    streamed_tool_calls = True

            elif part["type"] == "updates":
                for node_name, update in part["data"].items():
                    if not isinstance(update, dict):
                        continue

                    completed_messages = update.get("messages", [])
                    if node_name == "Memory Agent":
                        for message in completed_messages:
                            if not isinstance(message, AIMessage):
                                continue

                            reasoning = message.additional_kwargs.get("reasoning")
                            if reasoning and not streamed_reasoning:
                                start_section("Memory Agent Thinking")
                                print(reasoning, end="", flush=True)

                            if message.tool_calls and not streamed_tool_calls:
                                start_section("Memory Agent Tool Call")
                                for index, tool_call in enumerate(
                                    message.tool_calls
                                ):
                                    if index:
                                        print()
                                    print(
                                        json.dumps(
                                            tool_call,
                                            ensure_ascii=False,
                                            default=str,
                                        ),
                                        end="",
                                        flush=True,
                                    )

                            if message.content and not streamed_content:
                                start_section("Memory Agent Response")
                                print(message.content, end="", flush=True)

                            if getattr(self, "print_raw", False):
                                raw_response_count += 1
                                start_section(
                                    "Memory Agent Raw Response "
                                    f"#{raw_response_count}"
                                )
                                print(
                                    json.dumps(
                                        message.model_dump(mode="json"),
                                        indent=2,
                                        ensure_ascii=False,
                                        default=str,
                                    ),
                                    flush=True,
                                )

                        active_tool_index = None
                        streamed_content = False
                        streamed_reasoning = False
                        streamed_tool_calls = False

                    elif node_name == "Tool Node":
                        for message in completed_messages:
                            if not isinstance(message, ToolMessage):
                                continue
                            start_section("Memory Agent Tool Result")
                            tool_name = message.name or message.tool_call_id
                            print(
                                f"{tool_name}: {message.content}",
                                end="",
                                flush=True,
                            )

        if active_section is not None:
            print()

        if final_state is None:
            raise RuntimeError("Memory Agent completed without producing final state.")

        return final_state
        


    def run(self, messages, current_frame=None):

        request_type = None
        original_request = None
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            if not isinstance(message.content, str):
                break
            request = message.content.lstrip()
            original_request = request
            if request.startswith("RETRIEVE REQUEST:"):
                request_type = "RETRIEVE"
            elif request.startswith("MUTATE REQUEST:"):
                request_type = "MUTATE"
            break

        self._reset_request_state(request_type)
        self._original_request = original_request
        # Runtime callers provide the observation they own.  The webcam fetch
        # remains only as a backwards-compatible fallback for standalone use.
        if current_frame is None:
            current_frame = get_live_frame()
        if not isinstance(current_frame, Mapping):
            raise TypeError(
                "current_frame must be a model-compatible image mapping"
            )
        print(colored(f"\nMemory Agent:", "white", "on_blue"))
        response = self.invoke_agent(
            messages,
            current_frame=dict(current_frame),
        )
        if getattr(self, "_retrieval_performed", False):
            final_messages = response.get("messages", [])
            if not final_messages:
                raise RuntimeError("Memory Agent produced no final response")
            self._finalize_semantic_trace(final_messages[-1].content)

        return response
