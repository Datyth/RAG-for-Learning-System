# RAG-for-Learning-System

A minimal, production-style Retrieval-Augmented Generation (RAG) system for querying local PDF documents with grounded answers.

## What This Project Does

This project ingests PDF files from a local `data/` directory, splits them into chunks, embeds them with a Hugging Face embedding model, stores them in a local Qdrant collection, and answers questions using retrieved context only.

The design stays intentionally simple and inspectable:

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
- [uv](https://docs.astral.sh/uv/)
- For Gemini only: a valid `GOOGLE_API_KEY`

## Quick Start

### 1. Install Dependencies

```bash
uv sync
cp .env.example .env
```

### 2. Add Your PDFs

Place your PDF files in the `data/` directory.

### 3. Ingest Documents

```bash
uv run rag ingest
```

To recreate the collection from scratch:

```bash
uv run rag ingest --recreate
```

### 4. Ask Questions

```bash
uv run rag ask "What is LoRA fine-tuning?"
```

## LLM Providers

The active provider is controlled by `settings.llm_provider` in `src/config.py` and can be overridden at runtime with the `LLM_PROVIDER` environment variable.

| Provider | Value | API key required |
| --- | --- | --- |
| Hugging Face local (default) | `hf_local` | No |
| Gemini | `gemini` | Yes (`GOOGLE_API_KEY`) |

<details>
<summary><strong>Local Hugging Face (default)</strong></summary>

Runs a Transformers text-generation pipeline locally through LangChain's `ChatHuggingFace`.

- No network or API key is required after model weights are cached.
- The first run downloads the model to `~/.cache/huggingface`.

Configurable via environment variables (or `src/config.py`):

| Setting | Env var | Default |
| --- | --- | --- |
| Model ID or local path | `HF_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Device | `HF_DEVICE` | `-1` (CPU); use `0`, `1`, … for CUDA |
| Max new tokens | — | `512` |
| dtype | — | `bfloat16` |
| Temperature | — | `0.1` (shared across providers) |

Example:

```bash
uv run rag ask "What is LoRA fine-tuning?"
```

For larger models, consider using a GPU with `hf_device=0` and a lower-precision `hf_dtype`.

</details>

<details>
<summary><strong>Gemini (optional)</strong></summary>

Set the provider and API key in `.env`:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-api-key-here
```

The Gemini model name is controlled by `gemini_model` in `src/config.py` and defaults to `gemini-2.5-flash`.

Examples:

```bash
uv run rag ask "What is LoRA fine-tuning?"
```

Or override the provider for a single command:

```bash
LLM_PROVIDER=gemini uv run rag ask "What is LoRA fine-tuning?"
```

</details>

## CLI Usage

### Ingest PDFs

```bash
uv run rag ingest
uv run rag ingest --recreate
```

### Ask Questions

```bash
uv run rag ask "What is LoRA fine-tuning?"
uv run rag ask "Summarize the alignment paper" --k 8
uv run rag ask "What does page 3 say?" -f filename="[Reading]-LLM-Alignment.pdf" -f page=3
```

### Inspect Retrieval Without Calling the LLM

```bash
uv run rag debug-retrieval "reinforcement learning from human feedback"
uv run rag debug-retrieval "GPT pretraining" --k 10 --json
```

## Metadata Filtering

You can constrain retrieval with metadata filters when using `ask`.

Example:

```bash
uv run rag ask "What does page 3 say?" -f filename="[Reading]-LLM-Alignment.pdf" -f page=3
```

This is useful when you want to scope the answer to a specific file, page, or other indexed metadata field.

## Configuration

Application settings are defined in `src/config.py`. Default values include:

- Data directory: `./data`
- Qdrant storage directory: `./storage/qdrant`
- Collection name: `rag_chunks`
- Chunk size: `1000`
- Chunk overlap: `150`
- Embedding model: `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`
- LLM provider: `hf_local`
- HF model: `Qwen/Qwen3-4B-Instruct-2507`
- Gemini model: `gemini-2.5-flash`
- LLM temperature: `0.1`
- Default retrieval top-k: `5`