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


def load_documents(corpus_dir: str | Path, max_words: int = 80) -> list[DocumentChunk]:
    """Load Markdown paragraphs as small, metadata-preserving retrieval chunks."""
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        meta, lines = _split_front_matter(path.read_text(encoding="utf-8").splitlines())
        title = next((line.removeprefix("# ").strip() for line in lines if line.startswith("# ")), path.stem)
        paragraphs = "\n".join(lines).split("\n\n")
        chunk_id = 0
        for paragraph in paragraphs:
            text = " ".join(line.strip() for line in paragraph.splitlines() if not line.startswith("#"))
            words = text.split()
            for start in range(0, len(words), max_words):
                chunk_text = " ".join(words[start:start + max_words]).strip()
                if chunk_text:
                    chunks.append(DocumentChunk(chunk_text, f"{path.stem}-{chunk_id:03d}", path.name, title, chunk_id,
                                                 meta.get("updated", "")))
                    chunk_id += 1
    return chunks


def load_knowledge_base(corpus_root: str | Path, knowledge_base: str, max_words: int = 80) -> list[DocumentChunk]:
    """kb_0 is the generator's memory snapshot; kb_a/kb_b are the deployed KBs."""
    if knowledge_base not in {"kb_0", "kb_a", "kb_b"}:
        raise ValueError("knowledge_base must be 'kb_0', 'kb_a' or 'kb_b'")
    return load_documents(Path(corpus_root) / knowledge_base, max_words=max_words)
