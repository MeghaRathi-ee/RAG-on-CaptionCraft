from models.caption_model import CaptionGenerator
from retriever import Retriever

class RAGCaptionCraft:
    def __init__(self):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()

    def style_transform(self, caption, context):
        # Simple Instagram styling logic
        styled = caption.replace("a woman", "coffee vibes")
        return styled + " ☕✨"

    def generate(self, image_path):
        base_caption = self.captioner.generate_caption(image_path)

        context = self.retriever.retrieve(base_caption, k=1)[0]

        final_caption = self.style_transform(base_caption, context)

        return base_caption, final_caption, context
