"""Smoke tests: examples import and can build their agent graphs.

These do not call the LLM, so they don't need real API keys.
"""

from __future__ import annotations

import os


def _set_dummy_keys() -> None:
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(key, "test-key")


def test_basic_builds() -> None:
    _set_dummy_keys()
    from examples import basic

    agent = basic.build_agent()
    assert agent is not None


def test_subagents_builds() -> None:
    _set_dummy_keys()
    from examples import subagents

    agent = subagents.build_agent()
    assert agent is not None
