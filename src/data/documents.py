from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import random

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
    seed: int = 7,
) -> tuple[list[Query], list[Query]]:
    """Align KB-A and KB-B chunks by source sequence and produce paired queries.

    - queries_a: DIRECT always correct (memorized == current, no drift).
    - queries_b: DIRECT correct only for unchanged chunks; wrong for drifted ones.

    Changed chunk counts are supported. Equal chunks are anchors; changed regions
    are paired positionally. Unmatched KB-A chunks represent deletions, while
    KB-B-only additions have no frozen KB-A answer and are not paired.
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

    def align_source(source: str) -> list[tuple[DocumentChunk, DocumentChunk | None]]:
        source_a = a_by_src.get(source, [])
        source_b = b_by_src.get(source, [])
        matcher = SequenceMatcher(
            None,
            [chunk.text for chunk in source_a],
            [chunk.text for chunk in source_b],
            autojunk=False,
        )
        pairs = []
        for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
            if tag == "insert":
                continue
            b_region = source_b[b_start:b_end]
            for offset, chunk_a in enumerate(source_a[a_start:a_end]):
                chunk_b = b_region[offset] if offset < len(b_region) else None
                pairs.append((chunk_a, chunk_b))
        return pairs

    queries_a: list[Query] = []
    queries_b: list[Query] = []

    templates = [
        "What does the documentation say about {first_words}?",
        "Answer this from the docs: {first_words}.",
        "Look up the documentation details starting with {first_words}.",
        "Which documented rule begins with {first_words}?",
        "Can you identify the documentation entry about {topic_words}?",
    ]
    rng = random.Random(seed)
    global_index = 0
    for source in all_sources:
        for chunk_a, chunk_b in align_source(source):
            query_id = f"doc-q-{global_index:03d}"
            first_words = " ".join(chunk_a.text.split()[:8]).rstrip(".,")
            topic_words = " ".join(chunk_a.text.split()[:5]).rstrip(".,")
            template = rng.choice(templates)
            query_text = template.format(first_words=first_words, topic_words=topic_words)

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
                entity=chunk_b.source if chunk_b else chunk_a.source,
                attribute=chunk_b.title if chunk_b else chunk_a.title,
                text=query_text,
                memorized_answer=chunk_a.text,   # frozen KB-A knowledge
                current_answer=chunk_b.text if chunk_b else "",
                affected_by_drift=chunk_b is None or chunk_a.text != chunk_b.text,
            ))

            global_index += 1

    return queries_a, queries_b