from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document_id: str
    source: str
    title: str
    chunk_id: int
    updated: str = ""   # page's ISO "updated:" date from front matter; "" if absent
    knowledge_base: str = ""  # snapshot the chunk was loaded from (kb_0 / kb_a / kb_b); "" for ad-hoc corpora

    @property
    def value(self) -> str:
        return self.text


def _split_front_matter(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse a leading '---' block of 'key: value' lines."""
    if not lines or lines[0].strip() != "---":
        return {}, lines
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    meta = dict(line.split(":", 1) for line in lines[1:end] if ":" in line)
    return {key.strip(): value.strip() for key, value in meta.items()}, lines[end + 1:]


def _paragraph_blocks(lines: list[str]) -> list[str]:
    """Blank-line paragraphs with heading lines removed.

    A paragraph ending in ':' is a lead-in ("Create a user with:") for the
    block that follows it (e.g. "`POST /v1/users`"); the two are one
    statement, so they form one block. Lead-ins never join across a heading.
    """
    blocks, join_next = [], False
    for paragraph in "\n".join(lines).split("\n\n"):
        paragraph_lines = paragraph.splitlines()
        if any(line.startswith("#") for line in paragraph_lines):
            join_next = False
        text = " ".join(line.strip() for line in paragraph_lines if not line.startswith("#")).strip()
        if not text:
            continue
        if join_next:
            blocks[-1] = f"{blocks[-1]} {text}"
        else:
            blocks.append(text)
        join_next = text.endswith(":")
    return blocks


def load_documents(corpus_dir: str | Path, max_words: int = 80, knowledge_base: str = "") -> list[DocumentChunk]:
    """Load Markdown paragraphs as small, metadata-preserving retrieval chunks."""
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        meta, lines = _split_front_matter(path.read_text(encoding="utf-8").splitlines())
        title = next((line.removeprefix("# ").strip() for line in lines if line.startswith("# ")), path.stem)
        chunk_id = 0
        for text in _paragraph_blocks(lines):
            words = text.split()
            for start in range(0, len(words), max_words):
                chunk_text = " ".join(words[start:start + max_words]).strip()
                if chunk_text:
                    chunks.append(DocumentChunk(chunk_text, f"{path.stem}-{chunk_id:03d}", path.name, title, chunk_id,
                                                 meta.get("updated", ""), knowledge_base))
                    chunk_id += 1
    return chunks


MEMORY_SNAPSHOT = "kb_0"
RETRIEVABLE_SNAPSHOTS = ("kb_a", "kb_b")


def load_knowledge_base(corpus_root: str | Path, knowledge_base: str, max_words: int = 80) -> list[DocumentChunk]:
    """kb_0 is the generator's memory snapshot (never retrievable); kb_a/kb_b are the deployed KBs."""
    if knowledge_base not in {MEMORY_SNAPSHOT, *RETRIEVABLE_SNAPSHOTS}:
        raise ValueError("knowledge_base must be 'kb_0', 'kb_a' or 'kb_b'")
    return load_documents(Path(corpus_root) / knowledge_base, max_words=max_words, knowledge_base=knowledge_base)
