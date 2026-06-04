from __future__ import annotations

import faiss
import numpy as np

from retriever.core import INDEX_PATH, build_searchable_text, load_prompts, embed_text


def main() -> None:
    prompts = load_prompts()
    embeddings = []

    for prompt in prompts:
        text = build_searchable_text(prompt)
        embeddings.append(embed_text(text)[0])

    embeddings_array = np.array(embeddings, dtype="float32")
    # embed_text already normalises each vector; normalise the batch again to be safe
    faiss.normalize_L2(embeddings_array)
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatIP(dimension)  # inner product == cosine similarity on unit vectors
    index.add(embeddings_array)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Embeddings + index saved to {INDEX_PATH}.")


if __name__ == "__main__":
    main()
