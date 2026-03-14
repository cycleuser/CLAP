"""Configuration management for CLAP."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CLAPConfig:
    """CLAP configuration settings."""

    chat_model: str = "gemma3:1b"
    embed_model: str = "nomic-embed-text:latest"
    font_size: int = 12
    chunk_size: int = 2000
    chunk_overlap: int = 200
    persist_directory: str = ""

    def __post_init__(self):
        if not self.persist_directory:
            self.persist_directory = str(self.get_default_persist_dir())

    @staticmethod
    def get_default_persist_dir() -> Path:
        """Get default persistence directory."""
        config_home = Path.home() / ".clap"
        config_home.mkdir(parents=True, exist_ok=True)
        return config_home / "knowledge_base"

    @staticmethod
    def get_config_file() -> Path:
        """Get config file path."""
        config_home = Path.home() / ".clap"
        config_home.mkdir(parents=True, exist_ok=True)
        return config_home / "config.json"


def load_config() -> CLAPConfig:
    """Load configuration from file."""
    config_file = CLAPConfig.get_config_file()

    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)
            return CLAPConfig(**data)
        except Exception:
            pass

    return CLAPConfig()


def save_config(config: CLAPConfig) -> None:
    """Save configuration to file."""
    config_file = CLAPConfig.get_config_file()

    config_dict = asdict(config)
    config_dict["persist_directory"] = str(config.persist_directory)

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def is_first_run() -> bool:
    """Check if this is the first run."""
    return not CLAPConfig.get_config_file().exists()


def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        import ollama

        data = ollama.list()
        return [m["model"] for m in data.get("models", [])]
    except Exception:
        return []


def get_embedding_models() -> list:
    """Get list of embedding models."""
    all_models = get_available_models()
    embed_keywords = ["embed", "nomic", "bge", "e5", "mxbai", "snowflake", "arctic", "granite"]

    embed_models = []
    for model in all_models:
        model_lower = model.lower()
        if any(kw in model_lower for kw in embed_keywords):
            embed_models.append(model)

    if not embed_models:
        embed_models = [m for m in all_models if m]

    return embed_models


def get_chat_models() -> list:
    """Get list of chat models."""
    all_models = get_available_models()
    embed_keywords = ["embed", "nomic", "bge", "e5", "mxbai", "snowflake", "arctic", "granite"]

    chat_models = []
    for model in all_models:
        model_lower = model.lower()
        if not any(kw in model_lower for kw in embed_keywords):
            chat_models.append(model)

    if not chat_models:
        chat_models = all_models

    return chat_models
