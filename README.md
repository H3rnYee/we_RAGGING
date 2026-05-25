# we_RAGGING

A small RAG experiment for retrieving prompt strategies, boosting matches by tags,
and generating optimized prompts with Ollama.

## Setup

```bash
pip install -r requirements.txt
```

Make sure these Ollama models are available locally:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:1.7b
```

## Build Embeddings

Rebuild the FAISS index whenever `data/prompts.json` changes:

```bash
python -m embeddings.build_embeddings
```

## Run The Website

```bash
streamlit run app.py
```

The sidebar lets you choose tags to boost during retrieval. Vector similarity is
still used first, then selected tag matches rerank the candidate results.

## Run From The CLI

Retrieve and optimize a prompt:

```bash
python -m retriever.search
```

Compare raw and optimized model output:

```bash
python -m optimizer.optimize
```
