from __future__ import annotations

import ollama

from optimizer.optimizer_prompt import optimizer_template
from retriever.core import CHAT_MODEL, build_retrieved_context, generate_optimized_prompt
from retriever.core import search_prompts


def _parse_tags(raw_tags: str) -> list[str]:
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def main() -> None:
    user_prompt = input("Enter your prompt: ")
    raw_tags = input("Boost tags (comma-separated, optional): ")

    retrieved_results = search_prompts(user_prompt, tags=_parse_tags(raw_tags), k=10)
    retrieved_context = build_retrieved_context(retrieved_results)

    optimized_prompt = generate_optimized_prompt(
        user_prompt,
        retrieved_context,
        optimizer_template,
    )

    raw_output = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
    )["message"]["content"]

    optimized_output = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": optimized_prompt}],
    )["message"]["content"]

    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(user_prompt)

    print("\n" + "=" * 80)
    print("RETRIEVED STRATEGIES")
    print("=" * 80)
    for i, result in enumerate(retrieved_results, start=1):
        print(
            f"{i}. {result.prompt['content']} "
            f"(score={result.boosted_score:.3f}, tag_boost={result.tag_score:.3f})"
        )

    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPT")
    print("=" * 80)
    print(optimized_prompt)

    print("\n" + "=" * 80)
    print("RAW OUTPUT")
    print("=" * 80)
    print(raw_output)

    print("\n" + "=" * 80)
    print("OPTIMIZED OUTPUT")
    print("=" * 80)
    print(optimized_output)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        "Observe differences in structure, clarity, completeness, reasoning, "
        "error handling, formatting, and technical depth."
    )


if __name__ == "__main__":
    main()
