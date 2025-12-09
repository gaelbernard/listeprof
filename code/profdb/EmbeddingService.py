import os
import logging
import numpy as np
import lmdb
from openai import OpenAI
from more_itertools import chunked
import hashlib

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from dotenv import load_dotenv
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(Path(PROJECT_ROOT) / ".env")

def text_to_hash(text: str) -> str:
    """Generate a hash for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingService:
    def __init__(self, cache_path: str = None, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dtype = np.float64
        self.batch_size = 64  # OpenAI batch limit

        if cache_path:
            self.cache = lmdb.open(cache_path, map_size=1_000_000_000)
        else:
            self.cache = None

    def embed(self, texts) -> np.ndarray:
        """
        Embed one or more texts.

        Args:
            texts: A single string or list of strings

        Returns:
            np.ndarray of shape (n_texts, embedding_dim) if list input,
            or (embedding_dim,) if single string input
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = [None] * len(texts)
        texts_to_embed = []  # (index, text) pairs for texts not in cache

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_embed.append((i, text))

        # Embed uncached texts in batches
        for chunk in chunked(texts_to_embed, self.batch_size):
            indices = [item[0] for item in chunk]
            chunk_texts = [item[1] for item in chunk]

            resp = self.client.embeddings.create(input=chunk_texts, model=self.model)

            for idx, text, data in zip(indices, chunk_texts, resp.data):
                emb = np.array(data.embedding, dtype=self.dtype)
                embeddings[idx] = emb
                self._save_to_cache(text, emb)

        result = np.array(embeddings)
        return result[0] if single_input else result

    def _load_from_cache(self, text: str):
        """Load embedding from cache, or return None if not found."""
        if self.cache is None:
            return None
        hash_key = text_to_hash(text)
        with self.cache.begin(write=False) as txn:
            raw = txn.get(hash_key.encode("utf-8"))
            if raw is None:
                return None
            return np.frombuffer(raw, dtype=self.dtype)

    def _save_to_cache(self, text: str, emb: np.ndarray):
        """Save embedding to cache."""
        if self.cache is None:
            return
        hash_key = text_to_hash(text)
        with self.cache.begin(write=True) as txn:
            txn.put(hash_key.encode("utf-8"), emb.tobytes())