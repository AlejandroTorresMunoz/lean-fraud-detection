"""Backend factory for the fraud-triage agent: the single place that picks the chat model.

`agent.provider` in the config selects the LLM, and nothing else in the agent knows which one runs:

- ``ollama`` — a local, $0, offline model via `langchain-ollama`'s ChatOllama (tool-calling +
  structured output). The default for real use.
- ``mock`` — a deterministic in-memory fake chat model for tests/CI. Tests inject a scripted list of
  AIMessages (tool calls, then a final answer) via ``responses``; the default is a harmless empty
  reply so a plain ``provider: mock`` run never hits the network.

The whole LangChain stack (including langchain-ollama) is installed by default (`uv sync`), so both
imports are safe at module load and CI reproduces this environment without extra flags.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


def build_chat_model(cfg: dict, responses: list | None = None) -> BaseChatModel:
    """Return the chat model selected by ``cfg['agent']['provider']`` (``ollama`` | ``mock``)."""
    agent_cfg = cfg["agent"]
    provider = agent_cfg["provider"]
    if provider == "ollama":
        # base_url lets the API reach an Ollama running in another container (OLLAMA_BASE_URL, e.g.
        # http://ollama:11434); unset -> ChatOllama's localhost default for host runs.
        base_url = os.getenv("OLLAMA_BASE_URL") or agent_cfg.get("base_url")
        return ChatOllama(
            model=agent_cfg.get("model", "qwen2.5:3b"),
            temperature=agent_cfg.get("temperature", 0.0),
            base_url=base_url,
        )
    if provider == "mock":
        return FakeMessagesListChatModel(responses=responses or [AIMessage(content="")])
    raise ValueError(f"Unknown agent.provider {provider!r} in config. Expected 'ollama' or 'mock'.")
