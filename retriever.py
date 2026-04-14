from pathlib import Path
from uuid import uuid4

from chromadb import Client
from chromadb.config import Settings

from models.embedding_model import EmbeddingModel


KNOWLEDGE_PATH = Path("data/knowledge/instagram_style.txt")
HASHTAG_PATH = Path("data/knowledge/instagram_hashtags.txt")


class Retriever:
    """
    Handles retrieval for style examples and hashtags, and also stores
    optional user-specific style captions for personalization.
    """

    def __init__(self):
        self.client = Client(
            Settings(
                persist_directory="vector_store",
                anonymized_telemetry=False,
            )
        )

        # Knowledge base collection for generic style captions
        self.collection = self.client.get_or_create_collection(
            name="captioncraft_kb"
        )

        # Separate collection for user-specific captions
        self.user_collection = self.client.get_or_create_collection(
            name="captioncraft_user"
        )

        self.embedder = EmbeddingModel()

        self._ensure_index_built()

    def _ensure_index_built(self) -> None:
        """
        Lazily build the vector index from the Instagram style
        knowledge file if the Chroma collection is currently empty.
        """
        if self.collection.count() > 0:
            return

        if not KNOWLEDGE_PATH.exists():
            return

        with KNOWLEDGE_PATH.open(encoding="utf-8") as f:
            documents = [line.strip() for line in f if line.strip()]

        if not documents:
            return

        embeddings = self.embedder.encode(documents)
        ids = [f"kb-{i}" for i in range(len(documents))]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
        )

    @staticmethod
    def _strip_tags(text: str) -> str:
        # Remove leading tags like "[coffee]" or "[user:alice]"
        if text.startswith("[") and "]" in text:
            return text.split("]", 1)[1].strip()
        return text

    def add_user_caption(self, username: str, caption: str) -> None:
        """
        Add a user-specific caption to the user collection so that
        future generations can be nudged towards their personal style.
        """
        if not username or not caption:
            return

        tagged = f"[user:{username}] {caption}"
        emb = self.embedder.encode([tagged]).tolist()

        self.user_collection.add(
            ids=[f"user-{username}-{uuid4()}"],
            documents=[tagged],
            embeddings=emb,
        )

    def retrieve(
        self,
        query: str,
        k: int = 3,
        style: str | None = None,
        username: str | None = None,
    ):
        """
        Retrieve style examples conditioned on the base caption, an
        optional style tag, and optionally include a user-specific
        caption if available.
        """
        if not query:
            return [""]

        q_emb = self.embedder.encode([query]).tolist()

        # Query generic style knowledge first
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=max(k * 3, k),
        )

        docs = results.get("documents", [[]])[0]

        # Apply simple client-side style filtering using tag prefixes
        if style:
            tag_prefix = f"[{style.lower()}]"
            styled_docs = [
                d for d in docs if d.lower().startswith(tag_prefix)
            ]
            if len(styled_docs) >= k:
                docs = styled_docs

        top_docs = docs[:k]

        # Optionally include one user-specific caption at the front
        if username:
            user_results = self.user_collection.query(
                query_embeddings=q_emb,
                n_results=1,
            )
            user_docs = user_results.get("documents", [[]])[0]
            if user_docs:
                top_docs = user_docs[:1] + top_docs

        if not top_docs:
            return [""]

        return [self._strip_tags(d) for d in top_docs]

    def retrieve_hashtags(self, query: str, style: str | None = None) -> str:
        """
        Lightweight hashtag retrieval: embed the query and compare
        against a small set of pre-defined hashtag bundles.
        """
        if not HASHTAG_PATH.exists():
            return ""

        with HASHTAG_PATH.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return ""

        # If a style is chosen, prefer lines tagged with that style
        if style:
            tag_prefix = f"[{style.lower()}]"
            styled = [l for l in lines if l.lower().startswith(tag_prefix)]
            if styled:
                lines = styled

        query_emb = self.embedder.encode([query]).tolist()
        cand_embs = self.embedder.encode(lines)

        # Cosine similarity for a small set is cheap to compute manually
        import numpy as np

        qv = np.array(query_emb[0])
        cvs = np.array(cand_embs)

        sims = (cvs @ qv) / (
            np.linalg.norm(cvs, axis=1) * np.linalg.norm(qv) + 1e-8
        )

        best_idx = int(np.argmax(sims))
        best = lines[best_idx]

        return self._strip_tags(best)
