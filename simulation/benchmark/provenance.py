"""Stable provenance helpers shared by benchmark evaluation runners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_DIRECTORIES = (
    "agents",
    "memory",
    "dataset",
    "simulation/benchmark",
    "prompt",
)
_SOURCE_SUFFIXES = frozenset({".py", ".md", ".json"})


def _hash_file_set(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def source_tree_sha256(repository_root: str | Path) -> str:
    """Hash agent, memory, benchmark, and prompt sources used by evaluation."""

    root = Path(repository_root).resolve()
    paths: list[Path] = []
    main = root / "main.py"
    if main.is_file():
        paths.append(main)
    for relative in _SOURCE_DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            continue
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in _SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    return _hash_file_set(root, paths)


def runtime_provenance(repository_root: str | Path) -> dict[str, Any]:
    """Return runtime identity plus a hash that captures uncommitted sources."""

    root = Path(repository_root).resolve()
    commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        candidate = completed.stdout.strip().casefold()
        if completed.returncode == 0 and _GIT_COMMIT.fullmatch(candidate):
            commit = candidate
    except (OSError, subprocess.SubprocessError):
        pass

    packages: dict[str, str | None] = {}
    for distribution in ("ollama", "openai", "Pillow"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "git_commit": commit,
        "source_tree_sha256": source_tree_sha256(root),
        "packages": packages,
    }


def benchmark_content_sha256(
    benchmark_root: str | Path,
    scenario_ids: Iterable[str],
) -> str:
    """Hash every selected manifest and frame, not only scenario identifiers."""

    root = Path(benchmark_root).resolve()
    paths: list[Path] = []
    for scenario_id in sorted(map(str, scenario_ids)):
        episode = root / "episodes" / scenario_id
        paths.extend(
            (
                episode / "manifest.json",
                episode / "1.png",
                episode / "2.png",
            )
        )
    return _hash_file_set(root, paths)


def callable_provenance(value: Any) -> dict[str, str | None]:
    """Return a stable identifier and source hash for an injected hook."""

    module = str(getattr(value, "__module__", "")).strip()
    qualname = str(
        getattr(value, "__qualname__", getattr(value, "__name__", ""))
    ).strip()
    identifier = (
        f"{module}.{qualname}".strip(".") or type(value).__qualname__
    )
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        code = getattr(value, "__code__", None)
        source = (
            repr(
                (
                    getattr(code, "co_code", None),
                    getattr(code, "co_consts", None),
                    getattr(code, "co_names", None),
                )
            ).encode("utf-8")
            if code is not None
            else b""
        )
    return {
        "id": identifier,
        "source_sha256": (
            hashlib.sha256(source).hexdigest() if source else None
        ),
    }


__all__ = [
    "benchmark_content_sha256",
    "callable_provenance",
    "runtime_provenance",
    "source_tree_sha256",
]
