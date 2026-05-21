import json
import ollama
import numpy as np
import faiss

# Load prompts
with open("./data/prompts.json", "r") as f:
    prompts = json.load(f)

texts = []

# Build searchable text
for prompt in prompts:
    combined_text = f"""
    Prompt Type: {prompt.get('prompt_type', '')}
    Tags: {' '.join(prompt.get('tags', []))}
    Content: {prompt.get('content', '')}
    """

    texts.append(combined_text)

# Generate embeddings
embeddings = []

for text in texts:
    response = ollama.embeddings(
        model='qwen3-embedding:0.6b',
        prompt=text
    )

    embeddings.append(response['embedding'])

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "./embeddings/faiss.index")

print("Embeddings + index saved.")
