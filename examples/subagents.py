"""02 — Deep agent with specialized sub-agents.

The orchestrator delegates to:
  * `researcher` — runs Tavily searches and returns raw findings.
  * `writer`     — turns those findings into a structured report.

This is where the "deep" in deepagents starts paying off: each role has a
focused system prompt and a narrow tool surface, which dramatically improves
quality versus a single generalist agent.

Run:
    uv run python -m examples.subagents
    uv run python -m examples.subagents "compare GPT-5 and Claude Sonnet 4.5" --stream
"""

from __future__ import annotations

from deepagents import SubAgent, create_deep_agent

from deep_agents_app import get_model
from deep_agents_app.tools import web_query
from examples._runner import parse_query, run

ORCHESTRATOR_PROMPT = """You coordinate a research workflow.

Workflow:
  1. Call the `researcher` sub-agent with the user's question to gather sources.
  2. Call the `writer` sub-agent with those findings to produce the final report.
  3. Return the writer's report to the user verbatim.

Do not perform searches yourself — always delegate to `researcher`.
"""

RESEARCHER_PROMPT = """You are a research sub-agent.

For the given question:
  * Run 1–3 `web_query` calls covering different angles.
  * Return a bulleted list of the most important facts, each with its source URL.
  * Do not editorialize. No prose paragraphs.
"""

WRITER_PROMPT = """You are a writing sub-agent.

You receive bulleted research notes. Produce a structured report with:
  * A one-sentence summary at the top.
  * 2–4 short sections with descriptive headings.
  * A `Sources` section listing every URL cited in the notes.

Do not invent facts; only use what's in the notes.
"""

DEFAULT_QUERY = "Compare the current world records for the men's and women's 100m sprint."

researcher: SubAgent = {
    "name": "researcher",
    "description": "Searches the web with Tavily and returns raw findings with source URLs.",
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [web_query],
}

writer: SubAgent = {
    "name": "writer",
    "description": "Turns bulleted research notes into a structured report. Has no tools.",
    "system_prompt": WRITER_PROMPT,
    "tools": [],
}


def build_agent():
    return create_deep_agent(
        model=get_model(),
        tools=[web_query],
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=[researcher, writer],
    )


def main() -> None:
    args = parse_query(DEFAULT_QUERY)
    run(build_agent(), args.query, stream=args.stream)


if __name__ == "__main__":
    main()
