import os
import chromadb
from chromadb.config import Settings
from models.embedding_model import EmbeddingModel

# ---------------------------------------------------------------------------
# ChromaDB persistent client
# ---------------------------------------------------------------------------
client = chromadb.PersistentClient(path="vector_store")

# Delete and recreate to avoid duplicate IDs on re-runs
try:
    client.delete_collection(name="captioncraft")
except Exception:
    pass

collection = client.create_collection(name="captioncraft")
embedder = EmbeddingModel()

# ---------------------------------------------------------------------------
# Load all .txt files from data/knowledge/
# File naming convention: <style>.txt  e.g. travel.txt, food.txt
# The filename (without extension) becomes the style metadata tag.
# ---------------------------------------------------------------------------
texts = []
ids = []
metadatas = []

knowledge_dir = "data/knowledge"

for i, filename in enumerate(os.listdir(knowledge_dir)):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(knowledge_dir, filename)
    style_tag = os.path.splitext(filename)[0].capitalize()  # e.g. "Travel"

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        continue

    texts.append(content)
    ids.append(f"doc_{i}_{style_tag}")
    metadatas.append({"style": style_tag, "source": filename})

if not texts:
    print("⚠️  No .txt files found in data/knowledge/. Add knowledge files first.")
else:
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"✅ ChromaDB index created with {len(texts)} documents:")
    for m in metadatas:
        print(f"   - {m['source']}  [style={m['style']}]")