import os
import logging
import numpy as np
import lmdb
from openai import OpenAI, BadRequestError
from more_itertools import chunked
import hashlib

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from dotenv import load_dotenv
from pathlib import Path

# Resolve .env from common locations (repo root, /app in Docker, cwd).
_HERE = Path(__file__).resolve().parent
for _candidate in (
    _HERE.parents[1] / ".env",  # repo root when file is code/profdb/...
    Path("/app/.env"),
    Path.cwd() / ".env",
    _HERE / ".env",
):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

# Qwen3-Embedding commonly truncates around 8k tokens; stay slightly under.
MAX_INPUT_TOKENS = 8000
# Fallback character cap when tiktoken is unavailable. ~2 chars/token is a
# pessimistic ratio for non-English / dense unicode, so this stays safe.
FALLBACK_MAX_CHARS = MAX_INPUT_TOKENS * 2

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dep
    tiktoken = None


def text_to_hash(text: str, model: str = "") -> str:
    """Generate a cache key that includes the model (avoids cross-model reuse)."""
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _resolve_base_url() -> str:
    base_url = os.getenv("RCP_BASE_URL")
    if base_url:
        return base_url.rstrip("/")
    emb_url = (os.getenv("RCP_EMBEDDINGS_URL") or "").rstrip("/")
    if emb_url.endswith("/embeddings"):
        return emb_url[: -len("/embeddings")]
    raise ValueError(
        "Set RCP_BASE_URL or RCP_EMBEDDINGS_URL "
        "(e.g. https://inference.rcp.epfl.ch/v1)"
    )


class EmbeddingService:
    def __init__(self, cache_path: str = None, model: str = None):
        api_key = os.getenv("RCP_API")
        if not api_key:
            raise ValueError("RCP_API is not set in the environment / .env")

        # RCP exposes an OpenAI-compatible embeddings API.
        self.client = OpenAI(api_key=api_key, base_url=_resolve_base_url())
        self.model = model or os.getenv("EMBED_MODEL") or "Qwen/Qwen3-Embedding-8B"
        self.dtype = np.float64
        # Smaller batches for large embedding models on shared inference.
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))

        if tiktoken is not None:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self._encoding = None

        if cache_path:
            # Qwen3-Embedding-8B is 4096-d (float64 ≈ 32 KiB/vector). With ~35k
            # pubs plus older OpenAI entries, 2 GiB fills up; default to 16 GiB.
            map_size = int(os.getenv("EMBED_CACHE_MAP_SIZE", str(16 * 1024**3)))
            self.cache = lmdb.open(cache_path, map_size=map_size)
        else:
            self.cache = None

    def _truncate(self, text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
        """Truncate a text so the embeddings endpoint never rejects it.

        Uses tiktoken when available for an approximate token count; otherwise
        falls back to a conservative character cap.
        """
        if not text:
            return text
        if self._encoding is not None:
            tokens = self._encoding.encode(text, disallowed_special=())
            if len(tokens) <= max_tokens:
                return text
            truncated = self._encoding.decode(tokens[:max_tokens])
            logging.warning(
                f"Truncated embedding input from {len(tokens)} to {max_tokens} tokens"
            )
            return truncated
        max_chars = max_tokens * 2
        if len(text) <= max_chars:
            return text
        logging.warning(
            f"Truncated embedding input from {len(text)} to {max_chars} chars "
            f"(tiktoken unavailable, using char-based fallback)"
        )
        return text[:max_chars]

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
        texts_to_embed = []  # (index, original_text, safe_text) tuples not in cache

        # Check cache first. Cache keys remain the *original* text (plus model)
        # so prior entries stay valid; we only truncate what we send to the API.
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_embed.append((i, text, self._truncate(text)))

        n_cached = len(texts) - len(texts_to_embed)
        n_to_compute = len(texts_to_embed)
        if len(texts) > 1:  # Don't log for single embeddings
            logging.info(f"Embeddings: {n_cached} from cache, {n_to_compute} to compute")

        # Embed uncached texts in batches
        for chunk in chunked(texts_to_embed, self.batch_size):
            indices = [item[0] for item in chunk]
            cache_keys = [item[1] for item in chunk]
            chunk_texts = [item[2] for item in chunk]

            resp = self._embed_batch_with_retry(chunk_texts)

            for idx, cache_key, data in zip(indices, cache_keys, resp.data):
                emb = np.array(data.embedding, dtype=self.dtype)
                embeddings[idx] = emb
                self._save_to_cache(cache_key, emb)

        result = np.array(embeddings)
        return result[0] if single_input else result

    def _embed_batch_with_retry(self, chunk_texts):
        """Call the embeddings endpoint, retrying with harder truncation if the
        server still complains about input length. This is a defensive net on
        top of the up-front truncation in `_truncate`.
        """
        try:
            return self.client.embeddings.create(input=chunk_texts, model=self.model)
        except BadRequestError as e:
            msg = str(e).lower()
            if "maximum input length" not in msg and "too long" not in msg:
                raise
            logging.warning(
                "Embeddings endpoint rejected batch for length; retrying with "
                "halved token budget."
            )
            safer = [self._truncate(t, max_tokens=MAX_INPUT_TOKENS // 2) for t in chunk_texts]
            return self.client.embeddings.create(input=safer, model=self.model)

    def _load_from_cache(self, text: str):
        """Load embedding from cache, or return None if not found."""
        if self.cache is None:
            return None
        hash_key = text_to_hash(text, self.model)
        with self.cache.begin(write=False) as txn:
            raw = txn.get(hash_key.encode("utf-8"))
            if raw is None:
                return None
            return np.frombuffer(raw, dtype=self.dtype)

    def _save_to_cache(self, text: str, emb: np.ndarray):
        """Save embedding to cache."""
        if self.cache is None:
            return
        hash_key = text_to_hash(text, self.model)
        with self.cache.begin(write=True) as txn:
            txn.put(hash_key.encode("utf-8"), emb.tobytes())
