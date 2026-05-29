"""Tiny helpers shared by example scripts: CLI parsing + streaming printer."""

from __future__ import annotations

import argparse
from typing import Any

from langgraph.graph.state import CompiledStateGraph


def parse_query(default: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default=default,
        help="The question to send to the agent.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate steps instead of only printing the final answer.",
    )
    return parser.parse_args()


def run(agent: CompiledStateGraph, query: str, *, stream: bool) -> None:
    """Invoke or stream the agent and print the result."""
    payload: dict[str, Any] = {"messages": [("user", query)]}

    if not stream:
        result = agent.invoke(payload)
        print(result["messages"][-1].content)
        return

    final_state: dict[str, Any] | None = None
    for chunk in agent.stream(payload, stream_mode="values"):
        final_state = chunk
        msg = chunk["messages"][-1]
        role = getattr(msg, "type", msg.__class__.__name__)
        print(f"--- {role} ---")
        print(getattr(msg, "content", msg))
        print()

    if final_state is not None:
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_state["messages"][-1].content)
