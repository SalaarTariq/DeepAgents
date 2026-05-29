"""01 — Basic deep agent: one Tavily web-search tool, single system prompt.

Run:
    uv run python -m examples.basic
    uv run python -m examples.basic "your question here" --stream
"""

from __future__ import annotations

from deepagents import create_deep_agent

from deep_agents_app import get_model
from deep_agents_app.tools import web_query
from examples._runner import parse_query, run

SYSTEM_PROMPT = "Act as a researcher and provide detailed information on the topic."

DEFAULT_QUERY = "search for world record of 100m sprint"


def build_agent():
    return create_deep_agent(
        model=get_model(),
        tools=[web_query],
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    args = parse_query(DEFAULT_QUERY)
    run(build_agent(), args.query, stream=args.stream)


if __name__ == "__main__":
    main()
