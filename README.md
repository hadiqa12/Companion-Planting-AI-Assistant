# Companion Planting Assistant

A small **local RAG** (retrieval-augmented generation) over a hand-curated dataset of companion planting facts in `plants.json`.

- **Retrieval (always local):** each plant entry is embedded with [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and stored in a NumPy file. Queries are matched by cosine similarity.
- **Generation (optional, remote):** the `ask` command sends the top retrieved chunks plus your question to a remote [LM Studio](https://lmstudio.ai/) server (OpenAI-compatible API) and prints a grounded answer. If that server is unreachable, the script falls back to retrieval-only output.

## Repository layout

| File | Purpose |
|------|---------|
| `plants.json` | Source data: companion-planting facts per plant. |
| `chunk_example.py` | Turns one plant record into a single text chunk for embedding. |
| `local_rag.py` | CLI: `build`, `query`, `ask`. |
| `companion_embeddings.npy` | Prebuilt embedding matrix (one row per plant chunk). |
| `companion_meta.json` | Metadata aligned with the embeddings (plant name, id, chunk text). |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for environment variables (real values go in a local `.env`, never committed). |

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Copy the env template and edit it locally:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

Open `.env` and (optionally) point `LMSTUDIO_BASE_URL` at your remote LM Studio server, e.g.:

```
LMSTUDIO_BASE_URL=
LMSTUDIO_MODEL=openai/gpt-oss-20b
LMSTUDIO_API_KEY=
```

`.env` is git-ignored. The default in code is `http://127.0.0.1:1234`, so the script also works if you run LM Studio on the same machine.

## Usage

### 1. Build the index

```bash
python local_rag.py build
```

Embeds every plant chunk and writes `companion_embeddings.npy` and `companion_meta.json`. The first run downloads the MiniLM model (~90 MB); afterwards it works offline.

### 2. Search (retrieval only)

```bash
python local_rag.py query "What grows well with tomatoes?"
```

Prints the top matching plant chunks with similarity scores. No LLM call.

### 3. Ask (retrieval + remote LM Studio answer)

```bash
python local_rag.py ask "What grows well with tomatoes?"
```

What it does:

1. Retrieves the top chunks locally (MiniLM).
2. Sends a grounded prompt — system instructions + the chunks as context + your question — to `LMSTUDIO_BASE_URL/v1/chat/completions`.
3. Prints the model's answer.

If the LM Studio server is unreachable, times out, or returns an error, the command prints a `[warning]` and **falls back** to printing the retrieved chunks (the same output as `query`). The script never crashes for that reason.

Useful flags:

- `-k 8` — pass more chunks as context.
- `--show-context` — print retrieved chunks before the answer.

## Notes

- **Embedding model**: MiniLM is small (384-dim, fast on CPU). If you change the embedding model, **re-run `build`** — old vectors are not compatible.
- **Remote model**: the `ask` flow uses LM Studio's OpenAI-compatible API. Any OpenAI-compatible local server (e.g. another LM Studio instance, llama.cpp's `server`, vLLM) should work by changing only the `.env` values.
- **Quoting questions on Windows**: keep the entire question (including punctuation like `?`) inside the quotes, e.g. `"what doesn't grow well beside tomatoes?"`.

## License

No license specified; treat this as a personal/learning project.
