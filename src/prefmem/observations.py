"""Camera and dataset observation sources with recorder integration."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from prefmem.agents.contracts import Observation
from prefmem.agents.vision import (
    CapturedFrame,
    capture_live_frame,
    image_media_type,
)
from prefmem.recording import ExperimentRecorder, NullRecorder


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    observation: Observation
    frame_path: Path | None
    source: str


class ObservationSource(Protocol):
    def capture(self, *, purpose: str) -> ObservationEnvelope: ...


def _resize_frame(
    frame: CapturedFrame,
    *,
    max_width: int = 640,
    max_height: int = 480,
) -> CapturedFrame:
    """Resize an image only when it exceeds the configured model bounds."""
    import cv2
    import numpy

    encoded = numpy.frombuffer(frame.image, dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the captured frame")
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    ok, payload = cv2.imencode(".jpg", resized)
    if not ok:
        raise ValueError("OpenCV could not encode the resized frame")
    return CapturedFrame(
        image=payload.tobytes(),
        media_type="image/jpeg",
        source=frame.source,
        captured_at=frame.captured_at,
    )


class _BaseObservationSource:
    def __init__(
        self,
        *,
        recorder: ExperimentRecorder | None = None,
        resize_images: bool = False,
    ) -> None:
        self.recorder = recorder or NullRecorder()
        self.resize_images = resize_images
        self._sequence = 0

    def _envelope(
        self,
        frame: CapturedFrame,
        *,
        purpose: str,
    ) -> ObservationEnvelope:
        if self.resize_images:
            frame = _resize_frame(frame)
        self._sequence += 1
        digest = hashlib.sha256(frame.image).hexdigest()
        observation = Observation(
            observation_id=f"obs-{uuid4().hex}",
            captured_at=frame.captured_at,
            sequence=self._sequence,
            image_block=frame.model_block,
            content_hash=digest,
        )
        frame_path = self.recorder.record_frame(
            frame.image,
            source=purpose,
            media_type=frame.media_type,
            metadata={
                "camera_source": frame.source,
                "observation_id": observation.observation_id,
                "sequence": self._sequence,
                "captured_at": frame.captured_at.isoformat(),
                "sha256": digest,
            },
        )
        return ObservationEnvelope(
            observation=observation,
            frame_path=frame_path,
            source=frame.source,
        )


class LiveCameraObservationSource(_BaseObservationSource):
    def __init__(
        self,
        snapshot_url: str,
        *,
        timeout_seconds: float = 5.0,
        recorder: ExperimentRecorder | None = None,
        resize_images: bool = False,
    ) -> None:
        super().__init__(recorder=recorder, resize_images=resize_images)
        self.snapshot_url = snapshot_url
        self.timeout_seconds = timeout_seconds

    def capture(self, *, purpose: str) -> ObservationEnvelope:
        frame = capture_live_frame(
            self.snapshot_url,
            timeout=self.timeout_seconds,
        )
        return self._envelope(frame, purpose=purpose)


class DatasetObservationSource(_BaseObservationSource):
    """Read ordered image files, holding on the final frame when exhausted."""

    def __init__(
        self,
        dataset: Path,
        *,
        recorder: ExperimentRecorder | None = None,
        resize_images: bool = False,
    ) -> None:
        super().__init__(recorder=recorder, resize_images=resize_images)
        self.dataset = Path(dataset)
        if not self.dataset.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset} does not exist.")
        self.files = sorted(
            (
                path
                for path in self.dataset.iterdir()
                if path.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
            ),
            key=lambda path: tuple(
                (0, int(part)) if part.isdigit() else (1, part.casefold())
                for part in re.split(r"(\d+)", path.name)
            ),
        )
        if not self.files:
            raise FileNotFoundError(
                f"No supported image files found in {self.dataset}."
            )
        self._index = 0

    def capture(self, *, purpose: str) -> ObservationEnvelope:
        index = min(self._index, len(self.files) - 1)
        path = self.files[index]
        self._index += 1
        payload = path.read_bytes()
        frame = CapturedFrame(
            image=payload,
            media_type=image_media_type(payload),
            source=str(path.resolve()),
            captured_at=datetime.now(UTC),
        )
        return self._envelope(frame, purpose=purpose)
