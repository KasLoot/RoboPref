

from pathlib import Path
import base64


def image_data_url(image: bytes) -> str:
        if image.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif image.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        elif image.startswith((b"GIF87a", b"GIF89a")):
            media_type = "image/gif"
        elif image.startswith(b"RIFF") and image[8:12] == b"WEBP":
            media_type = "image/webp"
        elif image.startswith(b"BM"):
            media_type = "image/bmp"
        elif image.startswith((b"II*\x00", b"MM\x00*")):
            media_type = "image/tiff"
        else:
            raise ValueError(
                "vLLM image input must be PNG, JPEG, GIF, WebP, BMP, or TIFF."
            )
        encoded = base64.b64encode(image).decode("ascii")
        return f"data:{media_type};base64,{encoded}"


def get_start_end_frames(args):
        if args.dataset:
            dataset_path = Path(args.dataset)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

            # sort the files in the dataset directory to ensure consistent ordering
            image_files = sorted(dataset_path.glob("*.png"))
            if not image_files:
                raise FileNotFoundError(f"No PNG files found in dataset path {dataset_path}. Convert to PNG or provide a valid dataset.")
            
            start_frame = {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_files[0].read_bytes())},
            }
            last_frame = {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_files[-1].read_bytes())},
            }

            return start_frame, last_frame