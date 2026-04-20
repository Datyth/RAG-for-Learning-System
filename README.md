# RAG-for-Learning-System

A minimal, production-style Retrieval-Augmented Generation (RAG) system for querying local PDF documents with grounded answers.

## Overview

This project ingests PDF files from a local `data/` directory, splits them into chunks, embeds them with a Hugging Face model, stores them in a local Qdrant collection, and answers questions using retrieved context only.

Key capabilities:

- PDF ingestion with metadata preservation
- Local vector storage with Qdrant
- Retrieval with optional metadata filtering
- Prompt rendering via Jinja2
- Grounded answer generation with inline source markers
- CLI for ingesting, querying, and debugging retrieval
- Two LLM providers: local Hugging Face (default) and Gemini (optional)

## Project Structure

```text
.
├── data/                  # Local PDF documents for ingestion
├── src/
│   ├── __init__.py
│   ├── cli.py             # CLI entrypoints: ingest, ask, debug-retrieval
│   ├── config.py          # Application settings
│   ├── indexing.py        # PDF loading, metadata, chunking
│   ├── rag.py             # Retrieval, prompting, answer generation
│   ├── schemas.py         # Pydantic schemas
│   ├── store.py           # Embeddings and Qdrant setup
│   └── prompts/
│       └── answer.jinja2  # Answer-generation prompt template
├── storage/
│   └── qdrant/            # Local Qdrant storage
├── .env.example
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

## Requirements

- Python 3.11+
- `uv` or `pip` with a virtual environment
- For Gemini only: a valid `GOOGLE_API_KEY`

## Quick Start

### 1. Install dependencies

#### Using `uv` (recommended)

```bash
uv sync
cp .env.example .env
```

<details>
<summary><strong>Using <code>pip</code> instead</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

</details>

### 2. Add documents

Place your PDF files in `./data/`.

### 3. Configure the runtime

All runtime configuration lives in `src/config.py`.

Edit the `settings = Settings(...)` block to change paths, models, chunking parameters, retrieval settings, or device placement.

The `.env` file is used for secrets only:

```env
GOOGLE_API_KEY=your-api-key-here
```

This key is required only when `llm_provider = "gemini"`.

### 4. Ingest documents

```bash
uv run rag ingest
```

To rebuild the collection from scratch:

```bash
uv run rag ingest --recreate
```

If you installed with `pip install -e .`, you can also run:

```bash
rag ingest
```

### 5. Ask questions

```bash
uv run rag ask "What is LoRA fine-tuning?"
```

## LLM Providers

The active provider is controlled by `llm_provider` in `src/config.py`.

| Provider | `llm_provider` value | Secret required |
| --- | --- | --- |
| Local Hugging Face (default) | `hf_local` | None |
| Gemini | `gemini` | `GOOGLE_API_KEY` in `.env` |

<details>
<summary><strong>Local Hugging Face (default)</strong></summary>

Runs a local Transformers `text-generation` pipeline through LangChain's `ChatHuggingFace`.

Relevant settings in `src/config.py`:

- `hf_model` — local model path or model identifier
- `hf_device` — `-1` for CPU, `0+` for a CUDA device index
- `hf_max_new_tokens` — maximum generated tokens
- `llm_temperature` — shared with Gemini

This provider does not require an API key.

</details>

<details>
<summary><strong>Gemini (optional)</strong></summary>

To use Gemini:

1. Set `llm_provider = "gemini"` in `src/config.py`
2. Add `GOOGLE_API_KEY` to `.env`

```env
GOOGLE_API_KEY=your-api-key-here
```

The Gemini model name is controlled by `gemini_model` in `src/config.py`.

</details>

## CLI Usage

### Ingest PDFs

```bash
uv run rag ingest
uv run rag ingest --recreate
```

### Ask questions

```bash
uv run rag ask "What is LoRA fine-tuning?"
uv run rag ask "Summarize the alignment paper" --k 8
uv run rag ask "What does page 3 say?" -f filename="[Reading]-LLM-Alignment.pdf" -f page=3
```

When installed with `pip install -e .`, you can replace `uv run rag` with `rag`.

If no relevant context is retrieved, the system returns:

```text
I don't have enough information in the provided context to answer.
```

### Inspect retrieval

```bash
uv run rag debug-retrieval "reinforcement learning from human feedback"
uv run rag debug-retrieval "GPT pretraining" --k 10 --json
```

## Notes

- `storage/qdrant/` is local on-disk state and is excluded from version control.
- `.env` is for secrets only.
- The CLI entrypoint is `rag`, registered through `pyproject.toml`.
