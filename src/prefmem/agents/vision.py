import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_LIVE_FRAME_URL = "http://127.0.0.1:1234/snapshot.jpg"


def image_media_type(image: bytes) -> str:
    """Return the MIME type of a supported image payload."""
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    if image.startswith(b"BM"):
        return "image/bmp"
    if image.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    raise ValueError(
        "vLLM image input must be PNG, JPEG, GIF, WebP, BMP, or TIFF."
    )


def image_data_url(image: bytes) -> str:
    media_type = image_media_type(image)
    encoded = base64.b64encode(image).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """One immutable image observation and its model-compatible representation."""

    image: bytes
    media_type: str
    source: str
    captured_at: datetime

    @property
    def model_block(self) -> dict[str, object]:
        return {
            "type": "image_url",
            "image_url": {"url": image_data_url(self.image)},
        }


def capture_live_frame(
    snapshot_url: str = DEFAULT_LIVE_FRAME_URL,
    *,
    timeout: float = 5.0,
) -> CapturedFrame:
    """Fetch the latest webcam snapshot while retaining its original bytes."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    request = Request(snapshot_url, headers={"Accept": "image/jpeg"})
    try:
        with urlopen(request, timeout=timeout) as response:
            image = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"Could not get a live frame from {snapshot_url}: HTTP {exc.code}."
        ) from exc
    except (URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(
            f"Could not get a live frame from {snapshot_url}: {reason}. "
            "Is the webcam streamer running?"
        ) from exc

    if not image:
        raise RuntimeError(
            f"The webcam streamer returned an empty frame from {snapshot_url}."
        )

    return CapturedFrame(
        image=image,
        media_type=image_media_type(image),
        source=snapshot_url,
        captured_at=datetime.now(UTC),
    )


def get_live_frame(
    snapshot_url: str = DEFAULT_LIVE_FRAME_URL,
    *,
    timeout: float = 5.0,
) -> dict[str, object]:
    """Fetch the latest webcam snapshot as a model-compatible image block."""
    return capture_live_frame(snapshot_url, timeout=timeout).model_block


def get_start_end_frames(args):
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        # Sort files to ensure consistent ordering.
        image_files = sorted(dataset_path.glob("*.png"))
        if not image_files:
            raise FileNotFoundError(
                f"No PNG files found in dataset path {dataset_path}. "
                "Convert to PNG or provide a valid dataset."
            )

        start_frame = {
            "type": "image_url",
            "image_url": {"url": image_data_url(image_files[0].read_bytes())},
        }
        last_frame = {
            "type": "image_url",
            "image_url": {"url": image_data_url(image_files[-1].read_bytes())},
        }

        return start_frame, last_frame
