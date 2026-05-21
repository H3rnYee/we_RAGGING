import json
import ollama
import numpy as np
import faiss

# Load prompts
with open("./data/prompts.json", "r") as f:
    prompts = json.load(f)

# Load FAISS index
index = faiss.read_index("./embeddings/faiss.index")

# User query
query = input("Enter your prompt: ")

# Embed query
response = ollama.embeddings(
    model='qwen3-embedding:0.6b',
    prompt=query
)

query_embedding = np.array(
    [response['embedding']]
).astype("float32")

# Search
distances, indices = index.search(query_embedding, k=3)

# Print results
print("\nRetrieved Prompt Strategies:\n")

for idx in indices[0]:
    print(prompts[idx]["content"])
    print("-" * 50)