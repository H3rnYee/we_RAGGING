from __future__ import annotations

import json
import re

import ollama

from retriever.core import CHAT_MODEL


_JUDGE_PROMPT = """You are an impartial evaluator of prompt quality.

Original prompt:
{original}

Optimized prompt:
{optimized}

Score the optimized prompt on three dimensions (1–5 each):
- specificity: how concrete and domain-specific it is vs. vague or generic
- actionability: how clearly it tells the AI what to produce and in what form
- faithfulness: how well it preserves the original intent without adding or removing goals

Respond with a JSON object only — no markdown fences, no extra text:
{{"specificity": <int>, "actionability": <int>, "faithfulness": <int>, "reasoning": "<one sentence>"}}
"""


def _heuristics(original: str, optimized: str) -> dict:
    def avg_word_len(text: str) -> float:
        words = text.split()
        return sum(len(w) for w in words) / len(words) if words else 0.0

    orig_awl = avg_word_len(original)
    opt_awl = avg_word_len(optimized)
    clarity_delta = ((opt_awl - orig_awl) / orig_awl * 100) if orig_awl else 0.0

    orig_tokens = len(original.split())
    opt_tokens = len(optimized.split())
    length_ratio = opt_tokens / orig_tokens if orig_tokens else 0.0

    return {"clarity_delta": clarity_delta, "length_ratio": length_ratio}


def score_optimization(
    original: str,
    optimized: str,
    model: str = CHAT_MODEL,
) -> dict:
    prompt = _JUDGE_PROMPT.format(original=original, optimized=optimized)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip()

    # strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"specificity": 0, "actionability": 0, "faithfulness": 0, "reasoning": raw}

    scores["heuristics"] = _heuristics(original, optimized)
    return scores
