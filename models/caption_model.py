from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)
from PIL import Image


class CaptionGenerator:
    def __init__(self):

        # 🔹 Image Captioning Model (BLIP)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # 🔹 Instruction-Following Text Model (FLAN-T5)
        self.text_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )

    # ✅ Step 1 — Generate Base Caption from Image
    def generate_caption(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")

        output = self.image_model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    # ✅ Step 2 — Rewrite Caption using Retrieved Context
    def generate_text(self, prompt):
        output = self.text_generator(
            prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8
        )

        return output[0]["generated_text"].strip()
