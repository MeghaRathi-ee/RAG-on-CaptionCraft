from models.caption_model import CaptionGenerator
from retriever import Retriever

class RAGCaptionCraft:
    def __init__(self):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()

    def generate(self, image_path):
        base_caption = self.captioner.generate_caption(image_path)

        retrieved_context = (
            "Instagram captions are short, casual, and emoji-friendly."
        )

        prompt = f"""
        Context: {retrieved_context}
        Image Description: {base_caption}
        Generate a stylish caption.
        """

        return prompt
