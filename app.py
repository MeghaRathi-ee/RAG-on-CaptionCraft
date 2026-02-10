from rag_pipeline import RAGCaptionCraft

rag = RAGCaptionCraft()
result = rag.generate("data/images/sample.jpg")
print(result)
