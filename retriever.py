from chromadb import Client
from chromadb.config import Settings
from models.embedding_model import EmbeddingModel

class Retriever:
    def __init__(self):
        self.client = Client(
            Settings(
                persist_directory="vector_store",
                anonymized_telemetry=False
            )
        )

        # 🔥 IMPORTANT: use get_or_create_collection
        self.collection = self.client.get_or_create_collection(name="captioncraft")

        self.embedder = EmbeddingModel()

    def retrieve(self, query, k=1):
        q_emb = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=k
        )

        return results["documents"][0]
