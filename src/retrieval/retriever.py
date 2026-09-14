from dataclasses import dataclass
import numpy as np
from .embeddings import Embedder

@dataclass
class RetrievalResult:
    fact: object
    score: float

class Retriever:
    def __init__(
        self,
        facts,
        embedder=None,
        use_faiss=False,
        use_semantic_embeddings=False,
        embedding_model="all-MiniLM-L6-v2",
        allow_embedding_fallback=True,
    ):
        self.embedder = embedder or Embedder(
            model_name=embedding_model,
            use_model=use_semantic_embeddings,
            allow_fallback=allow_embedding_fallback,
        )
        self.facts = list(facts)
        self.vectors = self.embedder.encode([fact.text for fact in self.facts]) if self.facts else np.empty((0, 1))
        self.index = None
        if use_faiss:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.vectors.shape[1])
                self.index.add(self.vectors)
            except Exception:
                pass

    def search(self, query, top_k=3):
        if not self.facts:
            return []
        query_text = query.text if hasattr(query, "text") else str(query)
        vector = self.embedder.encode([query_text])
        if self.index is not None:
            scores, indices = self.index.search(vector, min(top_k, len(self.facts)))
            pairs = zip(indices[0], scores[0])
        else:
            scores = (self.vectors @ vector[0]).tolist()
            pairs = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [RetrievalResult(self.facts[int(index)], float(score)) for index, score in pairs if int(index) >= 0]

    @property
    def embedding_metadata(self) -> dict:
        return self.embedder.metadata()
