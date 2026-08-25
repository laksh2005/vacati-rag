"""Two chunkers, one per content type. See README > Chunking strategy for the why.

Both return the same shape:
    {id, doc, title, section, text, embed_text, meta}

`text` is what we show the user; `embed_text` is what we embed (the heading
breadcrumb is prefixed so near-identical sections stay distinguishable).
`id` is a content hash, so re-ingesting only re-embeds what actually changed.
"""

import hashlib
import json
from pathlib import Path

MAX_TOKENS = 450   # split a section that is longer than this
MIN_TOKENS = 60    # merge a section shorter than this into the next one
OVERLAP_CHARS = 240  # ~60 tokens of overlap between split parts


def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for sizing chunks."""
    return max(1, len(text) // 4)


def _chunk_id(doc: str, section: str, text: str) -> str:
    raw = f"{doc}|{section}|{text}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _make(doc: str, title: str, section: str, text: str, meta: dict) -> dict:
    breadcrumb = f"{title} > {section}" if section else title
    return {
        "id": _chunk_id(doc, section, text),
        "doc": doc,
        "title": title,
        "section": breadcrumb,
        "text": text,
        "embed_text": f"{breadcrumb}\n{text}",
        "meta": meta,
    }


def _split_long(text: str) -> list[str]:
    """Pack paragraphs up to MAX_TOKENS, carrying a little overlap forward."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parts: list[str] = []
    current = ""
    for para in paragraphs:
        if current and est_tokens(current + para) > MAX_TOKENS:
            parts.append(current.strip())
            current = current[-OVERLAP_CHARS:] + "\n\n"
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(path: Path) -> list[dict]:
    """Split prose on markdown headings: one section is one topic or one policy."""
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem.replace("-", " ").title()
    sections: list[tuple[str, list[str]]] = []
    heading = ""

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            heading = line[3:].strip()
            sections.append((heading, []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip():
            sections.append(("", [line]))

    chunks: list[dict] = []
    carried = ""  # a too-short section waiting to be merged into the next one
    carried_headings: list[str] = []
    for heading, body in sections:
        text = (carried + "\n".join(body)).strip()
        if not text:
            continue
        headings = [h for h in [*carried_headings, heading] if h]
        if est_tokens(text) < MIN_TOKENS:
            carried, carried_headings = text + "\n\n", headings
            continue
        carried, carried_headings = "", []
        # A merged chunk keeps both headings, so the citation still says where it came from.
        section = " / ".join(headings)
        for part in _split_long(text):
            chunks.append(_make(path.name, title, section, part, {"type": "prose"}))

    if carried.strip() and chunks:  # trailing short section: attach to the last chunk
        chunks[-1]["text"] += "\n\n" + carried.strip()
        chunks[-1]["section"] += " / " + " / ".join(carried_headings)
    return chunks


def _render(record: dict) -> str:
    """Turn a record into a sentence, so it sits in the same space as the questions."""
    parts = []
    for key, value in record.items():
        if key == "id" or value in (None, "", []):
            continue
        label = key.replace("_", " ")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{label}: {value}")
    return f"[{record['id']}] " + ". ".join(parts) + "."


def chunk_records(path: Path) -> list[dict]:
    """One chunk per record, never split: a record is a single atomic fact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    title = data["title"]
    return [
        _make(
            path.name,
            title,
            f"{data['record_type']} {record['id']} {record['name']}",
            _render(record),
            {"type": data["record_type"], "record_id": record["id"], **{
                k: v for k, v in record.items() if isinstance(v, (int, float, str))
            }},
        )
        for record in data["records"]
    ]


def chunk_directory(data_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted((data_dir / "docs").glob("*.md")):
        chunks.append(chunk_markdown(path))
    for path in sorted((data_dir / "catalog").glob("*.json")):
        chunks.append(chunk_records(path))
    return [c for group in chunks for c in group]
