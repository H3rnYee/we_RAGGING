from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT_DIR / "data" / "prompts.json"
INDEX_PATH = ROOT_DIR / "embeddings" / "faiss.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.1-8b-instant"

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _groq_client() -> Groq:
    import os
    import streamlit as st
    api_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    return Groq(api_key=api_key)


@dataclass(frozen=True)
class SearchResult:
    prompt: dict
    distance: float
    similarity: float
    tag_score: float
    boosted_score: float
    matched_tags: list[str]


def load_prompts(path: Path = PROMPTS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: Path = INDEX_PATH) -> faiss.Index:
    return faiss.read_index(str(path))


def get_all_tags(prompts: Iterable[dict]) -> list[str]:
    tags = {
        tag
        for prompt in prompts
        for tag in prompt.get("tags", [])
        if tag != "all"
    }
    return sorted(tags)


def build_searchable_text(prompt: dict) -> str:
    return "\n".join(
        [
            f"Prompt Type: {prompt.get('prompt_type', '')}",
            f"Layer: {prompt.get('layer', '')}",
            f"Priority: {prompt.get('priority', '')}",
            f"Tags: {' '.join(prompt.get('tags', []))}",
            f"Content: {prompt.get('content', '')}",
        ]
    )


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    vec = _get_embedder().encode([text], normalize_embeddings=True).astype("float32")
    return vec  # shape (1, dim), already L2-normalised


def _tag_boost(prompt: dict, query_tags: set[str], tag_weight: float) -> tuple[float, list[str]]:
    prompt_tags = set(prompt.get("tags", []))
    matched_tags = sorted(prompt_tags & query_tags)

    if not query_tags:
        return 0.0, []

    coverage = len(matched_tags) / len(query_tags)
    universal_boost = 0.25 if "all" in prompt_tags else 0.0
    return tag_weight * (coverage + universal_boost), matched_tags


def search_prompts(
    query: str,
    *,
    tags: Iterable[str] | None = None,
    k: int = 5,
    candidate_k: int | None = None,
    tag_weight: float = 0.35,
    prompts: list[dict] | None = None,
    index: faiss.Index | None = None,
) -> list[SearchResult]:
    prompts = prompts or load_prompts()
    index = index or load_index()
    candidate_k = min(candidate_k or max(k * 4, k), len(prompts))
    selected_tags = {tag.strip() for tag in tags or [] if tag.strip()}

    query_embedding = embed_text(query)
    distances, indices = index.search(query_embedding, candidate_k)

    results: list[SearchResult] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        prompt = prompts[int(idx)]
        similarity = float(distance)  # inner product == cosine similarity after L2 normalisation
        tag_score, matched_tags = _tag_boost(prompt, selected_tags, tag_weight)
        priority_score = float(prompt.get("priority", 0)) / 1000
        boosted_score = similarity + tag_score + priority_score

        results.append(
            SearchResult(
                prompt=prompt,
                distance=float(distance),
                similarity=similarity,
                tag_score=tag_score,
                boosted_score=boosted_score,
                matched_tags=matched_tags,
            )
        )

    results.sort(key=lambda result: result.boosted_score, reverse=True)
    return results[:k]


def build_retrieved_context(results: Iterable[SearchResult]) -> str:
    return "\n".join(f"- {result.prompt['content']}" for result in results)


def _build_optimizer_message(
    user_prompt: str,
    retrieved_context: str,
    optimizer_template: str,
) -> list[dict]:
    final_prompt = optimizer_template.format(
        retrieved_context=retrieved_context,
        user_prompt=user_prompt,
    )
    return [{"role": "user", "content": final_prompt}]


def generate_optimized_prompt(
    user_prompt: str,
    retrieved_context: str,
    optimizer_template: str,
    model: str = CHAT_MODEL,
) -> str:
    messages = _build_optimizer_message(user_prompt, retrieved_context, optimizer_template)
    response = _groq_client().chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def stream_optimized_prompt(
    user_prompt: str,
    retrieved_context: str,
    optimizer_template: str,
    model: str = CHAT_MODEL,
):
    """Yield text chunks as the model generates them."""
    messages = _build_optimizer_message(user_prompt, retrieved_context, optimizer_template)
    stream = _groq_client().chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
