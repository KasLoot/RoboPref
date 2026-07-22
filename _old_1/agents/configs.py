import os
from pathlib import Path



class PrefMem_Global_Config:
    workspace_root: str = str(Path(__file__).resolve().parents[1])
    dataset_path: str = str(Path(workspace_root) / "dataset" / "v3")
    memory_store_path: str = str(Path(__file__).resolve().parents[1] / "memory" / "preferences.json")
    user_id: str = "default"
    resize_images: bool = False
    image_width: int = 640
    image_height: int = 480
    image_jpeg_quality: int = 85


class HRI_Agent_Config(PrefMem_Global_Config):
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    system_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompt" / "hri" / "prompt-v1.md")


class Planner_Agent_Config(PrefMem_Global_Config):
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    system_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompt" / "planner" / "prompt-v1.md")


class Memory_Agent_Config(PrefMem_Global_Config):
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    system_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompt" / "memory" / "prompt-v1.md")


class Validator_Agent_Config(PrefMem_Global_Config):
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    system_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompt" / "validator" / "prompt-v1.md")

