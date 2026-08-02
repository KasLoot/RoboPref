#!/usr/bin/env python3

# Set BLAS thread counts before importing NumPy.
import os

NUM_THREADS = os.environ.get("NUM_THREADS", str(os.cpu_count() or 1))

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, NUM_THREADS)

import argparse
import time
from pathlib import Path

import numpy as np


NUM_VECTORS = 1_000_000
DIMENSIONS = 768
DTYPE = np.float32
DEFAULT_CHUNK_SIZE = 10_000
FILE_PATH = Path("mem.npy")


def format_bytes(number_of_bytes: int) -> str:
    return f"{number_of_bytes / (1024 ** 3):.2f} GiB"


def create_embeddings_file(
    path: Path,
    chunk_size: int,
    seed: int = 12345,
) -> None:
    """
    Create normalized random embeddings directly in an .npy file.

    Chunked generation avoids temporarily allocating the entire matrix
    before saving it.
    """
    expected_bytes = NUM_VECTORS * DIMENSIONS * np.dtype(DTYPE).itemsize

    print("Creating embedding file")
    print(f"Shape:              ({NUM_VECTORS:,}, {DIMENSIONS})")
    print(f"Data type:          {np.dtype(DTYPE)}")
    print(f"Expected file size: {format_bytes(expected_bytes)}")
    print(f"Chunk size:         {chunk_size:,}")
    print()

    start_time = time.perf_counter()

    embeddings = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=DTYPE,
        shape=(NUM_VECTORS, DIMENSIONS),
    )

    rng = np.random.default_rng(seed)

    for start in range(0, NUM_VECTORS, chunk_size):
        end = min(start + chunk_size, NUM_VECTORS)
        current_size = end - start

        block = rng.standard_normal(
            size=(current_size, DIMENSIONS),
            dtype=DTYPE,
        )

        # Normalize every row so cosine similarity becomes a dot product.
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block /= np.maximum(norms, np.finfo(DTYPE).eps)

        embeddings[start:end] = block

        if end % 100_000 == 0 or end == NUM_VECTORS:
            elapsed = time.perf_counter() - start_time
            print(
                f"Generated {end:,}/{NUM_VECTORS:,} vectors "
                f"({100 * end / NUM_VECTORS:.0f}%) "
                f"in {elapsed:.2f} seconds"
            )

    embeddings.flush()
    del embeddings

    elapsed = time.perf_counter() - start_time

    print()
    print(f"Saved file:         {path.resolve()}")
    print(f"Creation time:      {elapsed:.3f} seconds")
    print(f"Actual file size:   {format_bytes(path.stat().st_size)}")
    print()


def benchmark_cosine_similarity(
    path: Path,
    repeats: int,
    seed: int = 67890,
) -> None:
    print("Loading embeddings into CPU RAM")

    load_start = time.perf_counter()

    # mmap_mode=None means the whole matrix is loaded into CPU memory.
    embeddings = np.load(path, mmap_mode=None)

    load_elapsed = time.perf_counter() - load_start

    if embeddings.shape != (NUM_VECTORS, DIMENSIONS):
        raise ValueError(
            f"Unexpected shape {embeddings.shape}; "
            f"expected ({NUM_VECTORS}, {DIMENSIONS})"
        )

    if embeddings.dtype != DTYPE:
        raise ValueError(
            f"Unexpected dtype {embeddings.dtype}; expected {DTYPE}"
        )

    print(f"Load time:          {load_elapsed:.3f} seconds")
    print(f"RAM used by array:  {format_bytes(embeddings.nbytes)}")
    print(f"C-contiguous:       {embeddings.flags.c_contiguous}")
    print(f"Requested threads:  {NUM_THREADS}")
    print()

    rng = np.random.default_rng(seed)

    temp_vector = rng.standard_normal(
        size=DIMENSIONS,
        dtype=DTYPE,
    )
    temp_vector /= np.linalg.norm(temp_vector)

    # Small warm-up to initialize BLAS/thread pools.
    _ = embeddings[:10_000] @ temp_vector

    print("Running exact cosine similarity")
    print("Operation: embeddings @ temp_vector")
    print()

    timings = []
    scores = None

    for run_number in range(1, repeats + 1):
        search_start = time.perf_counter()

        # Because both vectors and query are normalized, this dot product
        # is exactly cosine similarity.
        scores = embeddings @ temp_vector

        search_elapsed = time.perf_counter() - search_start
        timings.append(search_elapsed)

        vectors_per_second = NUM_VECTORS / search_elapsed
        bandwidth_gb_s = embeddings.nbytes / search_elapsed / 1_000_000_000

        print(
            f"Run {run_number}: "
            f"{search_elapsed:.6f} seconds | "
            f"{vectors_per_second:,.0f} vectors/second | "
            f"{bandwidth_gb_s:.2f} GB/s"
        )

    assert scores is not None

    top_k = 10
    top_start = time.perf_counter()

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    top_elapsed = time.perf_counter() - top_start

    print()
    print(f"Minimum search time: {min(timings):.6f} seconds")
    print(f"Average search time: {np.mean(timings):.6f} seconds")
    print(f"Median search time:  {np.median(timings):.6f} seconds")
    print(f"Top-{top_k} time:       {top_elapsed:.6f} seconds")
    print()

    print(f"Top {top_k} cosine similarities:")
    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank:2d}. document={index:7d}, "
            f"similarity={scores[index]:.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cosine similarity over one million embeddings."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate mem.npy even when it already exists.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of vectors generated per chunk.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of cosine-search benchmark runs.",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    if args.recreate or not FILE_PATH.exists():
        create_embeddings_file(
            path=FILE_PATH,
            chunk_size=args.chunk_size,
        )
    else:
        print(f"Using existing file: {FILE_PATH.resolve()}")
        print("Use --recreate to regenerate it.")
        print()

    benchmark_cosine_similarity(
        path=FILE_PATH,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()