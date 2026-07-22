from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_JPEG_QUALITY = 85


class VisionImageError(ValueError):
    """Raised when an image cannot be prepared for a vision request."""


def prepare_vision_image(
    path: str | Path,
    *,
    resize: bool = True,
    width: int = DEFAULT_IMAGE_WIDTH,
    height: int = DEFAULT_IMAGE_HEIGHT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Return an in-memory JPEG suitable for an Ollama vision request.

    When resizing is enabled, the complete frame is fitted inside ``width`` x
    ``height`` without cropping, stretching, or upscaling.
    """
    image_path = Path(path).expanduser().resolve()
    if resize and (width <= 0 or height <= 0):
        raise VisionImageError(
            f"Image resize dimensions must be positive for {image_path}: {width}x{height}"
        )
    if not 1 <= jpeg_quality <= 95:
        raise VisionImageError(
            f"JPEG quality must be between 1 and 95 for {image_path}: {jpeg_quality}"
        )

    try:
        metadata = image_path.stat()
    except OSError as error:
        raise VisionImageError(f"Could not access vision image {image_path}: {error}") from error

    return _prepare_vision_image_cached(
        str(image_path),
        metadata.st_mtime_ns,
        metadata.st_size,
        resize,
        width,
        height,
        jpeg_quality,
    )


@lru_cache(maxsize=16)
def _prepare_vision_image_cached(
    path: str,
    modified_at_ns: int,
    source_size: int,
    resize: bool,
    width: int,
    height: int,
    jpeg_quality: int,
) -> bytes:
    # modified_at_ns and source_size deliberately participate in the cache key.
    del modified_at_ns, source_size
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba_image = image.convert("RGBA")
                rgb_image = Image.new("RGB", rgba_image.size, "white")
                rgb_image.paste(rgba_image, mask=rgba_image.getchannel("A"))
                image = rgb_image
            else:
                image = image.convert("RGB")

            if resize:
                image.thumbnail((width, height), Image.Resampling.LANCZOS)

            output = BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
            return output.getvalue()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise VisionImageError(f"Could not prepare vision image {image_path}: {error}") from error
