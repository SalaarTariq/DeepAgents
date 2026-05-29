"""Shared configuration: env loading and model factory."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "groq:qwen/qwen3-32b"


def load_env() -> None:
    """Load `.env` once. Safe to call from every example."""
    load_dotenv()


def get_model(name: str | None = None) -> BaseChatModel:
    """Return a chat model. `name` uses LangChain's `provider:model` format.

    Examples:
        get_model()                          # default Groq model
        get_model("openai:gpt-5")
        get_model("anthropic:claude-sonnet-4-5-20250929")
    """
    load_env()
    model_name = name or os.getenv("DEEP_AGENT_MODEL", DEFAULT_MODEL)
    return init_chat_model(model_name)
