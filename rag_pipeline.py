from models.caption_model import CaptionGenerator
from retriever import Retriever
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class RAGCaptionCraft:
    def __init__(self):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()

        # Load BLIP again for conditioned generation
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def generate(self, image_path):
        # Step 1: Base caption
        base_caption = self.captioner.generate_caption(image_path)

        # Step 2: Retrieve context
        context = self.retriever.retrieve(base_caption, k=1)[0]

        # Step 3: Load image
        image = Image.open(image_path).convert("RGB")

        # Step 4: Create contextual prompt
        prompt = f"{context}\nImage description: {base_caption}\nStylish caption:"

        inputs = self.processor(image, text=prompt, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            max_new_tokens=25
        )

        final_caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return base_caption, final_caption, context
