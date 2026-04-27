"""
Local RAG index: embed each plant chunk with sentence-transformers, search by similarity.

First-time setup:
  pip install -r requirements.txt

Build index (downloads model once ~90MB, then works offline):
  python local_rag.py build

Search (prints top matching chunks, retrieval only):
  python local_rag.py query "What grows well with tomatoes?"

RAG answer using remote LM Studio for generation (configured in .env):
  python local_rag.py ask "What grows well with tomatoes?"

If the LM Studio server is unreachable, `ask` falls back to retrieval-only
output (the same as `query`).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunk_example import plant_to_chunk

# Load variables from a local .env file (if present) into os.environ.
# Safe to import even if python-dotenv is not installed yet.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_api_key(name: str) -> str:
    """Return the API key stored in the .env / environment under `name`.

    Raises a clear error if the variable is missing or empty so you don't
    accidentally call an LLM provider without a key.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing API key: set {name} in your .env file (see .env.example)."
        )
    return value


DIR = Path(__file__).resolve().parent
PLANTS_JSON = DIR / "plants.json"
EMBEDDINGS_NPY = DIR / "companion_embeddings.npy"
META_JSON = DIR / "companion_meta.json"

# Small, fast English model; 384 dimensions. Swap for a larger model if you need higher quality.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Remote LM Studio defaults (overridable via .env).
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "").strip()
LMSTUDIO_TIMEOUT_SECONDS = 600

RAG_SYSTEM_PROMPT = (
    "You are a companion-planting assistant. Answer the user's question using only "
    "the information in the provided plant notes. If the notes do not contain enough "
    "information, say so plainly. Be concise and practical."
)


def load_plants() -> list[dict]:
    with PLANTS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def build_index() -> None:
    plants = load_plants()
    meta = []
    chunks: list[str] = []
    for p in plants:
        chunk = plant_to_chunk(p)
        chunks.append(chunk)
        meta.append(
            {
                "plant": p["plant"],
                "id": p["id"],
                "chunk": chunk,
            }
        )

    print(f"Embedding {len(chunks)} plants with {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(EMBEDDINGS_NPY, embeddings.astype(np.float32))
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote {EMBEDDINGS_NPY} and {META_JSON}")


def search(query: str, top_k: int = 3) -> list[tuple[float, dict]]:
    if not EMBEDDINGS_NPY.is_file() or not META_JSON.is_file():
        raise SystemExit(
            "Index missing. Run: python local_rag.py build"
        )
    embeddings = np.load(EMBEDDINGS_NPY)
    with META_JSON.open(encoding="utf-8") as f:
        meta: list[dict] = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    q = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Cosine similarity = dot product when vectors are unit-normalized
    scores = embeddings @ q
    order = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), meta[i]) for i in order]


def build_rag_context(hits: list[tuple[float, dict]]) -> str:
    """Format retrieved chunks as a readable context block for the LLM."""
    parts: list[str] = []
    for i, (score, row) in enumerate(hits, start=1):
        parts.append(
            f"### Source {i} (plant: {row['plant']}, score: {score:.3f})\n{row['chunk']}"
        )
    return "\n\n".join(parts)


def lmstudio_chat(question: str, context: str) -> str:
    """Call the remote LM Studio server (OpenAI-compatible /v1/chat/completions).

    Raises ConnectionError on any network/HTTP problem so the caller can
    decide whether to fall back to retrieval-only output.
    """
    url = f"{LMSTUDIO_BASE_URL.rstrip('/')}/v1/chat/completions"
    user_content = (
        "Use only the following plant notes to answer.\n\n"
        f"{context}\n\nUser question: {question}"
    )
    payload = json.dumps(
        {
            "model": LMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if LMSTUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LMSTUDIO_API_KEY}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=LMSTUDIO_TIMEOUT_SECONDS) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ConnectionError(f"LM Studio at {LMSTUDIO_BASE_URL!r} unreachable: {e}") from e
    except urllib.error.HTTPError as e:
        raise ConnectionError(f"LM Studio returned HTTP error: {e}") from e

    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise ConnectionError(f"Unexpected LM Studio response: {data!r}") from e


def print_chunks(hits: list[tuple[float, dict]]) -> None:
    """Print retrieved chunks in the same format as the `query` subcommand."""
    for rank, (score, row) in enumerate(hits, start=1):
        print(f"\n--- #{rank}  score={score:.4f}  plant={row['plant']} ---\n")
        print(row["chunk"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local companion-planting embedding index")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Embed all plants and save index files")

    q = sub.add_parser("query", help="Search index by natural language")
    q.add_argument("text", help="Your question or keywords")
    q.add_argument("-k", "--top-k", type=int, default=3, help="How many chunks to return")

    a = sub.add_parser(
        "ask",
        help="Retrieve top chunks and answer with remote LM Studio (falls back to retrieval-only).",
    )
    a.add_argument("text", help="Your question")
    a.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="How many plant chunks to pass as context (default: 5)",
    )
    a.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks before the model answer",
    )

    args = parser.parse_args()
    if args.command == "build":
        build_index()
        return
    if args.command == "query":
        print_chunks(search(args.text, top_k=args.top_k))
        return
    if args.command == "ask":
        hits = search(args.text, top_k=args.top_k)
        if not hits:
            raise SystemExit("No matching chunks; try `python local_rag.py build`.")
        if args.show_context:
            print("--- Retrieved context ---")
            print_chunks(hits)
            print("\n--- Answer ---\n")

        context = build_rag_context(hits)
        try:
            answer = lmstudio_chat(args.text, context)
            print(answer)
        except ConnectionError as e:
            print(f"[warning] Remote LLM unavailable ({e}).")
            print("[warning] Falling back to retrieval-only output (MiniLM).\n")
            print_chunks(hits)
        return


if __name__ == "__main__":
    main()
