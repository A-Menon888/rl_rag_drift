import hashlib
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dimension=384, use_model=True, allow_fallback=True):
        self.dimension = dimension
        self.model_name = model_name
        self.requested_semantic_model = use_model
        self.allow_fallback = allow_fallback
        self.model = None
        self.load_error = None
        if use_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except Exception as error:
                self.load_error = error
                if not allow_fallback:
                    raise RuntimeError(
                        f"Unable to load requested sentence-transformers model '{model_name}'. "
                        "Install its dependencies/model or set allow_embedding_fallback: true."
                    ) from error

    @property
    def backend(self) -> str:
        return "sentence-transformers" if self.model is not None else "hashed-fallback"

    def metadata(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name if self.model is not None else None,
            "semantic_requested": self.requested_semantic_model,
            "fallback_allowed": self.allow_fallback,
            "fallback_active": self.model is None,
        }

    def encode(self, texts):
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                vectors[row, index] += 1
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors
