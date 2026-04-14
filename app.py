from rag_pipeline import RAGCaptionCraft

rag = RAGCaptionCraft()

result = rag.generate("data/images/sample.jpg")

print("\nBase Caption:", result["base_caption"])

print("\nRetrieved Context:")
for caption in result["retrieved_context"]:
    print("-", caption)

print("\nFinal RAG Caption:", result["final_caption"])
