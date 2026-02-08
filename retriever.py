import faiss
from models.embedding_model import EmbeddingModel

class Retriever:
    def __init__(self):
        self.index = faiss.read_index("vector_store/faiss_index.bin")
        self.embedder = EmbeddingModel()

    def retrieve(self, query, k=1):
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(q_emb, k)
        return idx
