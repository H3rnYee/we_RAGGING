from __future__ import annotations

"""
Provider abstraction for chat completions.

Resolution order:
  1. OPENAI_API_KEY env var → OpenAI (used on Streamlit Cloud / any remote host)
  2. OLLAMA_HOST or default localhost → Ollama (used locally)
"""

import os
from typing import Generator


def _use_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def chat(messages: list[dict], model: str | None = None, stream: bool = False):
    if _use_openai():
        return _openai_chat(messages, model=model or "gpt-4o-mini", stream=stream)
    return _ollama_chat(messages, model=model or "qwen3:1.7b", stream=stream)


def _openai_chat(messages: list[dict], model: str, stream: bool):
    from openai import OpenAI  # lazy import — not required locally

    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages, stream=stream)

    if stream:
        def _gen() -> Generator[str, None, None]:
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        return _gen()

    return response.choices[0].message.content


def _ollama_chat(messages: list[dict], model: str, stream: bool):
    import ollama

    if stream:
        def _gen() -> Generator[str, None, None]:
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                token = chunk["message"]["content"]
                if token:
                    yield token
        return _gen()

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]
