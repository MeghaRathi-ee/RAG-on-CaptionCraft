from rag_pipeline import RAGCaptionCraft

rag = RAGCaptionCraft()

base, final, context = rag.generate("data/images/sample.jpg")

print("\nBase Caption:", base)
print("\nRetrieved Context:", context)
print("\nFinal RAG Caption:", final)
