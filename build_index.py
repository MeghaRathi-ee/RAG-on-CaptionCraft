import os
from chromadb import Client
from chromadb.config import Settings
from models.embedding_model import EmbeddingModel

# Create persistent Chroma client
client = Client(
    Settings(
        persist_directory="vector_store",
        anonymized_telemetry=False
    )
)

collection = client.get_or_create_collection(name="captioncraft")

embedder = EmbeddingModel()

texts = []
ids = []

for i, file in enumerate(os.listdir("data/knowledge")):
    with open(f"data/knowledge/{file}", "r", encoding="utf-8") as f:
        texts.append(f.read())
        ids.append(f"doc_{i}")

embeddings = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids
)

print("✅ ChromaDB index created successfully")
