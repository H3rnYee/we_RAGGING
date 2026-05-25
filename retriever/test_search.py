from __future__ import annotations

from retriever.core import search_prompts


def _parse_tags(raw_tags: str) -> list[str]:
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def main() -> None:
    query = input("Enter your prompt: ")
    raw_tags = input("Boost tags (comma-separated, optional): ")

    print("\nRetrieved Prompt Strategies:\n")
    for result in search_prompts(query, tags=_parse_tags(raw_tags), k=3):
        prompt = result.prompt
        print(
            f"[{prompt.get('id', 'unknown')}] "
            f"score={result.boosted_score:.3f} "
            f"tag_boost={result.tag_score:.3f}"
        )
        print(prompt["content"])
        print("-" * 50)


if __name__ == "__main__":
    main()
