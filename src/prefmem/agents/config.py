from langchain.tools import tool
from langchain.chat_models import init_chat_model
import os
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from IPython.display import Image, display
from pathlib import Path
from pydantic import BaseModel, Field


PROMPT_ROOT = Path(__file__).resolve().parent / "prompt"



class VLLMConfig:
    def __init__(self):
        self.model = "/workspace/models/gemma-4-26B-A4B-it"
        # self.model = "/workspace/models/Qwen3.6-35B-A3B"
        self.model_base_url = "http://localhost:8000/v1"

class OllamaConfig:
    def __init__(self):
        self.model = "gemma4:31b-cloud"
        self.model_base_url = "https://ollama.com/api"


class Global_Model_Config:
    def __init__(
        self,
        model_config,
        *,
        model=None,
        model_base_url=None,
    ):

        if model_config == "vllm":
            self.model_config = VLLMConfig()
        elif model_config == "ollama":
            self.model_config = OllamaConfig()
        else:
            raise ValueError(f"Unsupported model configuration: {model_config}")
        self.model = model or self.model_config.model
        self.model_base_url = model_base_url or self.model_config.model_base_url



class HRI_Config(Global_Model_Config):
    def __init__(self, model_config, *, model=None, model_base_url=None):
        super().__init__(
            model_config,
            model=model,
            model_base_url=model_base_url,
        )
        
        self.system_prompt = (PROMPT_ROOT / "hri" / "hri-prompt-v5.md").read_text(
            encoding="utf-8"
        )


class Planner_Config(Global_Model_Config):
    def __init__(self, model_config, *, model=None, model_base_url=None):
        super().__init__(
            model_config,
            model=model,
            model_base_url=model_base_url,
        )
        
        self.system_prompt = (
            PROMPT_ROOT / "planner" / "planner-prompt-v2.md"
        ).read_text(encoding="utf-8")


class Monitor_Config(Global_Model_Config):
    def __init__(self, model_config, *, model=None, model_base_url=None):
        super().__init__(
            model_config,
            model=model,
            model_base_url=model_base_url,
        )
        self.system_prompt = (
            PROMPT_ROOT / "monitor" / "monitor-prompt-v1.md"
        ).read_text(encoding="utf-8")


class Memory_Config(Global_Model_Config):
    def __init__(self, model_config, *, model=None, model_base_url=None):
        super().__init__(
            model_config,
            model=model,
            model_base_url=model_base_url,
        )
        
        self.system_prompt = (
            PROMPT_ROOT / "memory" / "memory-prompt-v3.md"
        ).read_text(encoding="utf-8")

        self.embedding_model = "/workspace/models/embeddinggemma-300m"
        self.embedding_model_base_url = "http://127.0.0.1:8080/v1"
        self.embedding_dimensions = 768
        # AB-MEM-Q's frozen development/held-out selection chose q=1, k=3,
        # cap=3.  Keep the production candidate window aligned with that
        # measured setting so downstream semantic filtering is not exposed to
        # seven additional, uncalibrated candidates.
        self.top_k = 3
        self.top_cap_k = 3
