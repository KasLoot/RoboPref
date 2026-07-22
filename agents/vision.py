from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


class VisionImageError(ValueError):
    pass


def prepare_vision_image(
    path: str | Path,
    *,
    resize: bool = False,
    width: int = 640,
    height: int = 480,
    jpeg_quality: int = 85,
) -> bytes:
    """Return model-ready image bytes without needless quality loss.

    With resizing disabled, the validated source bytes are returned unchanged.  This
    is important for geometric/contact validation and also preserves PNG losslessly.
    Resizing is the only mode that re-encodes to JPEG.
    """
    image_path = Path(path).expanduser().resolve()
    if resize and (width <= 0 or height <= 0):
        raise VisionImageError("Resize dimensions must be positive.")
    if not 1 <= jpeg_quality <= 95:
        raise VisionImageError("JPEG quality must be between 1 and 95.")
    try:
        metadata = image_path.stat()
    except OSError as error:
        raise VisionImageError(f"Could not access {image_path}: {error}") from error
    return _prepare_cached(
        str(image_path),
        metadata.st_mtime_ns,
        metadata.st_size,
        resize,
        width,
        height,
        jpeg_quality,
    )


@lru_cache(maxsize=16)
def _prepare_cached(
    path: str,
    modified_at_ns: int,
    source_size: int,
    resize: bool,
    width: int,
    height: int,
    jpeg_quality: int,
) -> bytes:
    del modified_at_ns, source_size
    try:
        if not resize:
            with Image.open(path) as source:
                source.verify()
            return Path(path).read_bytes()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                image = rgb
            else:
                image = image.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
            return output.getvalue()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise VisionImageError(f"Could not prepare {path}: {error}") from error
