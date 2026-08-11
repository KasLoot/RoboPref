"""Client for the local SAM 3.1 open-vocabulary detection service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import ipaddress
import math
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np


class SamServiceError(RuntimeError):
    """The detector was unavailable or returned an invalid response."""


@dataclass(frozen=True, slots=True)
class SamDetection:
    object_id: int
    box_xyxy: tuple[int, int, int, int]
    mask_area: int
    mask: np.ndarray | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.object_id, bool) or not isinstance(self.object_id, int):
            raise ValueError("object_id must be an integer")
        if len(self.box_xyxy) != 4 or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in self.box_xyxy
        ):
            raise ValueError("box_xyxy must contain four integers")
        x1, y1, x2, y2 = self.box_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("box_xyxy must have positive exclusive bounds")
        if (
            isinstance(self.mask_area, bool)
            or not isinstance(self.mask_area, int)
            or self.mask_area <= 0
        ):
            raise ValueError("mask_area must be a positive integer")
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.ndim != 2:
                raise ValueError("mask must be a two-dimensional array")
            mask = mask.copy()
            mask.setflags(write=False)
            object.__setattr__(self, "mask", mask)
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(self.score)
                or not 0 <= self.score <= 1
            ):
                raise ValueError("score must be between zero and one")
            object.__setattr__(self, "score", float(self.score))


class OpenVocabularyDetector(Protocol):
    def detect(
        self,
        rgb: np.ndarray,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> tuple[SamDetection, ...]: ...


class SamExchangeRecorder(Protocol):
    """Fail-closed observer for the exact SAM HTTP image exchange."""

    def start(
        self,
        *,
        prompt: str,
        threshold: float,
        image_jpeg: bytes,
    ) -> object: ...

    def end(
        self,
        token: object,
        *,
        status_code: int,
        response_body: bytes,
    ) -> None: ...

    def error(
        self,
        token: object,
        *,
        error: BaseException,
        status_code: int | None,
        response_body: bytes | None,
    ) -> None: ...


def _decode_mask(value: object, *, height: int, width: int) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SamServiceError("mask_png_base64 must be a non-empty string")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise SamServiceError("mask_png_base64 is not valid base64") from error
    encoded = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise SamServiceError("mask_png_base64 is not a valid PNG image")
    if decoded.shape != (height, width):
        raise SamServiceError(
            "detector mask dimensions do not match the submitted image"
        )
    return decoded > 0


class Sam3Client:
    """Synchronous, strict client for the forwarded local SAM endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9000",
        *,
        threshold: float = 0.5,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        exchange_recorder: SamExchangeRecorder | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        if parsed.hostname.casefold() != "localhost":
            try:
                address = ipaddress.ip_address(parsed.hostname)
            except ValueError as error:
                raise ValueError("SAM base_url must use a loopback host") from error
            if not address.is_loopback:
                raise ValueError("SAM base_url must use a loopback host")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        self.base_url = base_url.rstrip("/")
        self.threshold = float(threshold)
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=float(timeout), trust_env=False)
        self._exchange_recorder = exchange_recorder

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def detect(
        self,
        rgb: np.ndarray,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> tuple[SamDetection, ...]:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be a uint8 HxWx3 image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        actual_threshold = self.threshold if threshold is None else float(threshold)
        if not 0 <= actual_threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
        if not ok:
            raise SamServiceError("could not encode the detector input image")
        image_jpeg = encoded.tobytes()
        exchange_token = (
            None
            if self._exchange_recorder is None
            else self._exchange_recorder.start(
                prompt=prompt.strip(),
                threshold=actual_threshold,
                image_jpeg=image_jpeg,
            )
        )
        response: httpx.Response | None = None
        try:
            response = self._client.post(
                f"{self.base_url}/detect",
                data={"prompt": prompt.strip(), "threshold": str(actual_threshold)},
                files={"image": ("frame.jpg", image_jpeg, "image/jpeg")},
            )
            response.raise_for_status()
            payload = response.json()
            detections = self._parse_response(payload, image.shape[1], image.shape[0])
        except (httpx.HTTPError, ValueError, SamServiceError) as error:
            if self._exchange_recorder is not None:
                self._exchange_recorder.error(
                    exchange_token,
                    error=error,
                    status_code=None if response is None else response.status_code,
                    response_body=None if response is None else response.content,
                )
            if isinstance(error, SamServiceError):
                raise
            raise SamServiceError(f"SAM detection request failed: {error}") from error
        if self._exchange_recorder is not None:
            self._exchange_recorder.end(
                exchange_token,
                status_code=response.status_code,
                response_body=response.content,
            )
        return detections

    @staticmethod
    def _parse_response(
        payload: object,
        expected_width: int,
        expected_height: int,
    ) -> tuple[SamDetection, ...]:
        if not isinstance(payload, Mapping):
            raise SamServiceError("SAM response must be a JSON object")
        image = payload.get("image")
        if not isinstance(image, Mapping):
            raise SamServiceError("SAM response must describe its image dimensions")
        if image.get("width") != expected_width or image.get("height") != expected_height:
            raise SamServiceError("SAM response dimensions do not match submitted image")
        raw_detections = payload.get("detections")
        if isinstance(raw_detections, (str, bytes)) or not isinstance(
            raw_detections, Sequence
        ):
            raise SamServiceError("SAM detections must be an array")
        if payload.get("count") != len(raw_detections):
            raise SamServiceError("SAM count does not match its detections array")
        detections: list[SamDetection] = []
        for index, raw in enumerate(raw_detections):
            if not isinstance(raw, Mapping):
                raise SamServiceError(f"detection {index} must be an object")
            allowed = {
                "object_id",
                "box_xyxy",
                "mask_area",
                "mask_png_base64",
                "score",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise SamServiceError(
                    f"detection {index} contains unknown fields: "
                    f"{', '.join(sorted(str(item) for item in unknown))}"
                )
            box = raw.get("box_xyxy")
            if isinstance(box, (str, bytes)) or not isinstance(box, Sequence):
                raise SamServiceError(f"detection {index} has an invalid box")
            try:
                box_tuple = tuple(int(value) for value in box)
                detection = SamDetection(
                    object_id=raw.get("object_id"),
                    box_xyxy=box_tuple,
                    mask_area=raw.get("mask_area"),
                    mask=_decode_mask(
                        raw.get("mask_png_base64"),
                        height=expected_height,
                        width=expected_width,
                    ),
                    score=raw.get("score"),
                )
            except (TypeError, ValueError) as error:
                raise SamServiceError(f"invalid detection {index}: {error}") from error
            x1, y1, x2, y2 = detection.box_xyxy
            if x2 > expected_width or y2 > expected_height:
                raise SamServiceError(f"detection {index} lies outside the image")
            if detection.mask is not None:
                actual_area = int(detection.mask.sum())
                if actual_area != detection.mask_area:
                    raise SamServiceError(
                        f"detection {index} mask area does not match its mask"
                    )
            detections.append(detection)
        return tuple(detections)


__all__ = [
    "OpenVocabularyDetector",
    "SamExchangeRecorder",
    "Sam3Client",
    "SamDetection",
    "SamServiceError",
]
