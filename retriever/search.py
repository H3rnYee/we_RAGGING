from __future__ import annotations

from optimizer.optimizer_prompt import optimizer_template
from retriever.core import build_retrieved_context, generate_optimized_prompt, search_prompts


def _parse_tags(raw_tags: str) -> list[str]:
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def main() -> None:
    query = input("Enter your prompt: ")
    raw_tags = input("Boost tags (comma-separated, optional): ")

    results = search_prompts(query, tags=_parse_tags(raw_tags), k=3)
    retrieved_context = build_retrieved_context(results)

    print("\nRetrieved Prompt Strategies:\n")
    for result in results:
        prompt = result.prompt
        tags = ", ".join(prompt.get("tags", []))
        print(
            f"[{prompt.get('id', 'unknown')}] "
            f"score={result.boosted_score:.3f} "
            f"similarity={result.similarity:.3f} "
            f"tag_boost={result.tag_score:.3f} "
            f"tags={tags}"
        )
        print(prompt["content"])
        print("-" * 50)

    optimized_prompt = generate_optimized_prompt(
        query,
        retrieved_context,
        optimizer_template,
    )

    print("\nOptimized Prompt:\n")
    print(optimized_prompt)


if __name__ == "__main__":
    main()
