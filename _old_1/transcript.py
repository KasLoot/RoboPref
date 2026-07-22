from __future__ import annotations

import builtins
import re
import sys
import threading
import traceback as traceback_module
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TextIO


ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class _TeeStream:
    def __init__(self, terminal: TextIO, transcript: TextIO, lock: threading.RLock):
        self.terminal = terminal
        self.transcript = transcript
        self.lock = lock

    def write(self, text: str) -> int:
        with self.lock:
            self.terminal.write(text)
            self.transcript.write(ANSI_ESCAPE.sub("", text))
            self.transcript.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.transcript.flush()

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


class TerminalTranscript:
    """Mirror terminal output and typed input to an append-only text transcript."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._input = None
        self._lock = threading.RLock()

    def __enter__(self) -> TerminalTranscript:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._file.write(f"\n=== Session {datetime.now(timezone.utc).isoformat()} ===\n")

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._input = builtins.input
        sys.stdout = _TeeStream(self._stdout, self._file, self._lock)
        sys.stderr = _TeeStream(self._stderr, self._file, self._lock)
        builtins.input = self._recording_input
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None and exc_value is not None and self._file is not None:
            traceback_module.print_exception(exc_type, exc_value, traceback, file=self._file)
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._input is not None:
            builtins.input = self._input
        if self._file is not None:
            self._file.flush()
            self._file.close()

    def _recording_input(self, prompt: str = "") -> str:
        if self._input is None or self._file is None:
            raise RuntimeError("Transcript recorder is not active.")
        if prompt:
            print(prompt, end="", flush=True)
        value = self._input()
        with self._lock:
            self._file.write(value + "\n")
            self._file.flush()
        return value