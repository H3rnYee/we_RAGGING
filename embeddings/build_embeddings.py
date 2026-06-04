from __future__ import annotations

import faiss
import numpy as np

from retriever.core import EMBEDDING_MODEL, INDEX_PATH, build_searchable_text, load_prompts
from retriever.core import embed_text


def main() -> None:
    prompts = load_prompts()
    embeddings = []

    for prompt in prompts:
        text = build_searchable_text(prompt)
        embeddings.append(embed_text(text, model=EMBEDDING_MODEL)[0])

    embeddings_array = np.array(embeddings, dtype="float32")
    # embed_text returns normalised vectors; normalise again as a safety net
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatIP(dimension)  # inner product == cosine similarity on unit vectors
    index.add(embeddings_array)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Embeddings + index saved to {INDEX_PATH}.")


if __name__ == "__main__":
    main()
