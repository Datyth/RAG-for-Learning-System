# RAG-for-Learning-System

A minimal, production-style Retrieval-Augmented Generation (RAG) system for querying local PDF documents with grounded answers.

## Overview

This project ingests PDF files from a local `data/` directory, splits them into chunks, embeds them with a HuggingFace embedding model, stores them in a local Qdrant collection, and answers questions using retrieved context only.

The system is designed to stay simple, explicit, and inspectable:

- PDF ingestion with metadata preservation
- Local vector storage with Qdrant
- Retrieval with optional metadata filtering
- Prompt rendering via Jinja2
- Grounded answer generation with inline source markers
- CLI for ingesting, querying, and debugging retrieval

## Project Structure

```text
src/
  __init__.py
  cli.py
  config.py
  indexing.py
  rag.py
  schemas.py
  store.py
  prompts/
    answer.jinja2
data/
storage/
  qdrant/
.env.example
pyproject.toml
README.md
````

## Core Components

* `src/indexing.py`
  Loads PDFs, extracts page-level content, splits documents into chunks, and attaches stable metadata.

* `src/store.py`
  Configures embeddings, initializes the local Qdrant client, creates the collection, and exposes the vector store.

* `src/rag.py`
  Handles retrieval, prompt rendering, citation formatting, LLM invocation, and final answer assembly.

* `src/cli.py`
  Provides the command-line interface: `ingest`, `ask`, and `debug-retrieval`.

* `src/config.py`
  Centralizes runtime settings such as paths, chunking parameters, model names, and API key loading.

* `src/schemas.py`
  Defines structured metadata and response models with Pydantic.

## Requirements

* Python 3.11+
* [uv](https://docs.astral.sh/uv/)
* A valid Google API key for Gemini

## Installation

Install dependencies with `uv`:

```bash
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Then set your API key in `.env`:

```env
GOOGLE_API_KEY=your-api-key-here
```

## Configuration

Application settings are defined in `src/config.py`.

Default configuration includes:

* Data directory: `./data`
* Qdrant storage directory: `./storage/qdrant`
* Collection name: `rag_chunks`
* Chunk size: `1000`
* Chunk overlap: `150`
* Embedding model: `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`
* LLM model: `gemini-2.5-flash`
* Default retrieval top-k: `5`

Adjust these values in `src/config.py` if needed.

## Data Ingestion

Place PDF files in the `data/` directory, then run:

```bash
uv run rag ingest
```

To rebuild the collection from scratch:

```bash
uv run rag ingest --recreate
```

During ingestion, each PDF is:

1. Loaded page by page
2. Assigned document-level and page-level metadata
3. Split into chunks
4. Embedded and stored in Qdrant

## Asking Questions

Query the indexed documents with:

```bash
uv run rag ask "What is LoRA fine-tuning?"
```

Override the number of retrieved chunks:

```bash
uv run rag ask "Summarize the alignment paper" --k 8
```

Filter retrieval by metadata:

```bash
uv run rag ask "What does page 3 say?" -f filename="[Reading]-LLM-Alignment.pdf" -f page=3
```

If no relevant context is retrieved, the system returns:

```text
I don't have enough information in the provided context to answer.
```

## Inspecting Retrieval

To inspect retrieved chunks without calling the LLM:

```bash
uv run rag debug-retrieval "reinforcement learning from human feedback"
```

Return retrieval results as JSON:

```bash
uv run rag debug-retrieval "GPT pretraining" --k 10 --json
```