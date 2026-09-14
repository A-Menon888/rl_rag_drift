from dataclasses import dataclass
from pathlib import Path

from .generator import Query


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document_id: str
    source: str
    title: str
    chunk_id: int

    @property
    def value(self) -> str:
        return self.text


def load_documents(corpus_dir: str | Path, max_words: int = 80) -> list[DocumentChunk]:
    """Load Markdown paragraphs as small, metadata-preserving retrieval chunks."""
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((line.removeprefix("# ").strip() for line in lines if line.startswith("# ")), path.stem)
        paragraphs = "\n".join(lines).split("\n\n")
        chunk_id = 0
        for paragraph in paragraphs:
            text = " ".join(line.strip() for line in paragraph.splitlines() if not line.startswith("#"))
            words = text.split()
            for start in range(0, len(words), max_words):
                chunk_text = " ".join(words[start:start + max_words]).strip()
                if chunk_text:
                    chunks.append(DocumentChunk(chunk_text, f"{path.stem}-{chunk_id:03d}", path.name, title, chunk_id))
                    chunk_id += 1
    return chunks


def load_knowledge_base(corpus_root: str | Path, knowledge_base: str, max_words: int = 80) -> list[DocumentChunk]:
    if knowledge_base not in {"kb_a", "kb_b"}:
        raise ValueError("knowledge_base must be 'kb_a' or 'kb_b'")
    return load_documents(Path(corpus_root) / knowledge_base, max_words=max_words)


def generate_document_queries(chunks: list[DocumentChunk]) -> list[Query]:
    """Create deterministic factual queries from the documentation chunks."""
    queries = []
    for index, chunk in enumerate(chunks):
        first_words = " ".join(chunk.text.split()[:8]).rstrip(".,")
        queries.append(Query(
            query_id=f"doc-q-{index:03d}",
            entity=chunk.source,
            attribute=chunk.title,
            text=f"What does the documentation say about {first_words}?",
            answer_by_version={0: chunk.text},
            direct_answer="UNKNOWN",
            affected_by_drift=False,
        ))
    return queries