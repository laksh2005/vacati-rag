"""Build the index: chunk every document, embed it, write it to index/.

Usage:  python ingest.py
"""

import json
from pathlib import Path

import numpy as np

from app.chunker import chunk_directory, est_tokens
from app.config import settings
from app.gemini import embed_texts
from app.retriever import CHUNKS_FILE, VECTORS_FILE, normalize

DATA_DIR = Path("data")


def main() -> None:
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(exist_ok=True)

    chunks = chunk_directory(DATA_DIR)
    print(f"chunked {len({c['doc'] for c in chunks})} documents into {len(chunks)} chunks")

    vectors = normalize(np.array(embed_texts([c["embed_text"] for c in chunks], "RETRIEVAL_DOCUMENT"), dtype=np.float32))

    (index_dir / CHUNKS_FILE).write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    np.save(index_dir / VECTORS_FILE, vectors)

    tokens = sum(est_tokens(c["embed_text"]) for c in chunks)
    cost = tokens / 1_000_000 * settings.price_embed_input
    print(f"embedded ~{tokens} tokens at {settings.embed_dim} dims for about ${cost:.4f}")
    print(f"wrote {index_dir / CHUNKS_FILE} and {index_dir / VECTORS_FILE}")


if __name__ == "__main__":
    main()
