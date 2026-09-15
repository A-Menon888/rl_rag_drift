import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)
_EMBEDDER_CACHE = {}

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", dimension=384, use_model=True, allow_fallback=True):
        self.dimension = dimension
        self.model_name = model_name
        self.requested_semantic_model = use_model
        self.allow_fallback = allow_fallback
        self.model = None
        self.load_error = None
        self._text_cache = {}
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
                logger.warning(
                    "Unable to load sentence-transformers model '%s'; using hashed-fallback embeddings: %s",
                    model_name,
                    error,
                )

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
        texts = list(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in self._text_cache))
        if missing:
            if self.model is not None:
                new_vectors = np.asarray(
                    self.model.encode(missing, normalize_embeddings=True), dtype="float32"
                )
            else:
                new_vectors = np.zeros((len(missing), self.dimension), dtype="float32")
                for row, text in enumerate(missing):
                    for token in text.lower().split():
                        digest = hashlib.sha256(token.encode()).digest()
                        index = int.from_bytes(digest[:4], "little") % self.dimension
                        new_vectors[row, index] += 1
                    norm = np.linalg.norm(new_vectors[row])
                    if norm:
                        new_vectors[row] /= norm
            self._text_cache.update(zip(missing, new_vectors))
        return np.asarray([self._text_cache[text] for text in texts], dtype="float32")


def get_cached_embedder(model_name="all-MiniLM-L6-v2", dimension=384, use_model=True, allow_fallback=True):
    key = (model_name, dimension, use_model, allow_fallback)
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = Embedder(
            model_name=model_name,
            dimension=dimension,
            use_model=use_model,
            allow_fallback=allow_fallback,
        )
    return _EMBEDDER_CACHE[key]
