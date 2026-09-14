import hashlib
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dimension=256, use_model=False):
        self.dimension = dimension
        self.model = None
        if use_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except Exception:
                pass

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
