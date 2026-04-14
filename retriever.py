import chromadb
from models.embedding_model import EmbeddingModel


class Retriever:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="vector_store")
        self.collection = self.client.get_or_create_collection(name="captioncraft")
        self.embedder = EmbeddingModel()

    def retrieve(self, query: str, k: int = 3, style: str = None) -> list[str]:
        q_emb = self.embedder.encode([query]).tolist()

        where = {"style": style} if style and style != "General" else None

        try:
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=k,
                where=where,
            )
            docs = results.get("documents", [[]])[0]
            if docs:
                return docs
        except Exception:
            pass

        try:
            results = self.collection.query(
                query_embeddings=q_emb,
                n_results=k,
            )
            docs = results.get("documents", [[]])[0]
            if docs:
                return docs
        except Exception:
            pass

        return [
            "Instagram captions are short, punchy, and emoji-friendly.",
            "Use trending hashtags relevant to the image content.",
            "Keep it casual, relatable, and authentic.",
        ]