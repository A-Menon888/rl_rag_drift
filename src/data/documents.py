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


def generate_document_queries(
    kb_a_chunks: list[DocumentChunk],
    kb_b_chunks: list[DocumentChunk],
) -> tuple[list[Query], list[Query]]:
    """Align KB-A and KB-B chunks by (source, chunk_id) and produce paired query lists.

    - queries_a: DIRECT always correct (memorized == current, no drift).
    - queries_b: DIRECT correct only for unchanged chunks; wrong for drifted ones.

    Raises ValueError loudly if chunk counts per file differ between snapshots.
    """
    # Group by source filename for alignment check
    def by_source(chunks: list[DocumentChunk]) -> dict[str, list[DocumentChunk]]:
        result: dict[str, list[DocumentChunk]] = {}
        for c in chunks:
            result.setdefault(c.source, []).append(c)
        return result

    a_by_src = by_source(kb_a_chunks)
    b_by_src = by_source(kb_b_chunks)

    all_sources = sorted(set(a_by_src) | set(b_by_src))
    for source in all_sources:
        a_count = len(a_by_src.get(source, []))
        b_count = len(b_by_src.get(source, []))
        if a_count != b_count:
            raise ValueError(
                f"Chunk count mismatch for '{source}': "
                f"kb_a has {a_count} chunks, kb_b has {b_count}. "
                f"Inspect both files manually — a paragraph may have been added or removed."
            )

    queries_a: list[Query] = []
    queries_b: list[Query] = []

    global_index = 0
    for source in all_sources:
        for chunk_a, chunk_b in zip(a_by_src[source], b_by_src[source]):
            query_id = f"doc-q-{global_index:03d}"
            first_words = " ".join(chunk_a.text.split()[:8]).rstrip(".,")
            query_text = f"What does the documentation say about {first_words}?"

            queries_a.append(Query(
                query_id=query_id,
                entity=chunk_a.source,
                attribute=chunk_a.title,
                text=query_text,
                memorized_answer=chunk_a.text,
                current_answer=chunk_a.text,
                affected_by_drift=False,  # by construction on KB-A
            ))

            queries_b.append(Query(
                query_id=query_id,       # same ID — allows join across KB-A/KB-B
                entity=chunk_b.source,
                attribute=chunk_b.title,
                text=query_text,
                memorized_answer=chunk_a.text,   # frozen KB-A knowledge
                current_answer=chunk_b.text,      # live KB-B ground truth
                affected_by_drift=(chunk_a.text != chunk_b.text),
            ))

            global_index += 1

    return queries_a, queries_b