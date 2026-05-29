# DeepAgents

Hands-on experiments with the [`deepagents`](https://pypi.org/project/deepagents/) framework — a LangChain-based library for building agents that combine planning, file tools, sub-agents, and custom tools.

Each file under `examples/` illustrates one concept. Shared building blocks (env loading, model factory, reusable tools) live in `deep_agents_app/`.

## Layout

```
deep_agents_app/      shared building blocks
  config.py           env loading + model factory
  tools/web.py        Tavily web-search tool
examples/
  basic.py            01 — minimal single-tool deep agent
  subagents.py        02 — orchestrator + specialized sub-agents
```

## Setup

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env   # then fill in the API keys you actually use
```

You only need keys for the providers/tools your example touches — the loader does not require all of them.

## Run

```bash
# default query, final answer only
uv run basic

# custom query
uv run basic "what is the world record for the marathon"

# stream every step (planning, sub-agent calls, tool calls)
uv run subagents "compare GPT-5 and Claude Sonnet 4.5" --stream
```

## Choosing a model

Defaults to `groq:qwen/qwen3-32b`. Override per-run:

```bash
DEEP_AGENT_MODEL=openai:gpt-5 uv run subagents "..."
```

Format follows LangChain's `provider:model`.
