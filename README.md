# RAG-for-Learning-System

A minimal, production-style Retrieval-Augmented Generation (RAG) learning system over local PDF documents, with grounded answers, summarization, quiz generation, and flashcard generation.

## Overview

This project ingests PDF files from a local `data/` directory, splits them into chunks, embeds them with a Hugging Face model, stores them in a local Qdrant collection, and produces grounded study material using retrieved context only.

Key capabilities:

- PDF ingestion with metadata preservation
- Local vector storage with Qdrant
- Retrieval with optional metadata filtering
- Prompt rendering via Jinja2
- Grounded answer generation with inline source markers
- Grounded document summarization (single-shot or staged map/reduce)
- Structured multiple-choice quiz generation with citations
- Structured flashcard generation for study and review
- JSON and Markdown exports for quizzes, flashcards, and summaries
- CLI for ingest, query, debug, summarize, quiz, flashcards
- Two LLM providers: local Hugging Face (default) and Gemini (optional)

## Project Structure

```text
.
├── data/                  # Local PDF documents for ingestion
├── src/
│   ├── __init__.py
│   ├── cli.py             # CLI: ingest, ask, debug-retrieval, summarize, quiz, flashcards
│   ├── config.py          # Application settings
│   ├── export.py          # JSON/Markdown export for learning outputs
│   ├── indexing.py        # PDF loading, metadata, chunking
│   ├── learning.py        # Summarization, quiz, and flashcard generation
│   ├── rag.py             # Retrieval, prompting, answer generation
│   ├── schemas.py         # Pydantic schemas (chunks + learning outputs)
│   ├── store.py           # Embeddings and Qdrant setup
│   └── prompts/
│       ├── answer.jinja2
│       ├── summary_single.jinja2
│       ├── summary_map.jinja2
│       ├── summary_reduce.jinja2
│       ├── quiz.jinja2
│       └── flashcards.jinja2
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

## Learning Features

All learning commands reuse the indexed corpus, metadata filters, and citation model. Outputs are grounded in retrieved chunks only. If evidence is insufficient, generation returns fewer items or fails clearly instead of fabricating content.

Scoping options (shared across `summarize`, `quiz`, `flashcards`):

- `--document, -d FILENAME` — target a single indexed PDF (matches `metadata.filename`)
- `--query, -q TEXT` — topic- or question-guided retrieval, ranked by similarity
- `--filter, -f key=value` — any additional metadata filter (repeatable)
- `--k N` — retrieval `top_k` override for query mode
- `--output, -o PATH` — write output to a file instead of stdout
- `--format text|json|md` — stdout/file format (default `text`, renders Markdown-like text)

With no scope options, these commands run over the entire corpus.

### Summarize

Generates a grounded, study-oriented summary with key points and citations. For long inputs, the pipeline auto-falls back to a staged map/reduce summarization.

```bash
uv run rag summarize --document "[Reading]-LLM-Alignment.pdf"
uv run rag summarize --query "LoRA fine-tuning" --k 12
uv run rag summarize -d paper.pdf -o exports/paper-summary.md --format md
```

### Quiz generation

Generates a structured multiple-choice quiz set with answers, explanations, and source markers tied back to retrieved chunks.

```bash
uv run rag quiz --document "[Reading]-LLM-Alignment.pdf" --count 10
uv run rag quiz --query "reward model training" -n 6 --format json -o exports/rm-quiz.json
uv run rag quiz -f filename=paper.pdf -f page=3 -n 4
```

### Flashcard generation

Generates reusable flashcards suitable for spaced repetition, each grounded in the source.

```bash
uv run rag flashcards --document paper.pdf --count 20
uv run rag flashcards --query "PEFT methods" -n 12 --format md -o exports/peft-cards.md
```

### Output formats

- `text` — human-readable Markdown rendered to stdout or file
- `md` — Markdown file (when `--output` is set)
- `json` — stable, machine-friendly Pydantic JSON dump

JSON outputs carry the full schema (items/cards, citations, scope, target). Markdown outputs are easy to copy into notes or teaching materials.

## Notes

- `storage/qdrant/` is local on-disk state and is excluded from version control.
- `.env` is for secrets only.
- The CLI entrypoint is `rag`, registered through `pyproject.toml`.
- Learning defaults (`summarize_batch_size`, `summarize_retrieval_k`, `quiz_default_count`, `flashcards_default_count`, `generation_retrieval_k`) live in `src/config.py`.
