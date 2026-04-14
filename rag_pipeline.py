import json
import os
from dotenv import load_dotenv
from models.caption_model import CaptionGenerator
from retriever import Retriever
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"


class RAGCaptionCraft:

    def __init__(self):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()
        self.client = Groq(api_key=GROQ_API_KEY)

    def _call_groq(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Groq error: {str(e)}"

    def style_transform(self, caption: str, context: str, style: str) -> list[str]:
        prompt = f"""Write 3 Instagram captions for this image: "{caption}"
Style: {style}
Each caption on its own line. Include emojis and 3 hashtags. No numbering."""

        raw = self._call_groq(prompt)
        captions = [line.strip() for line in raw.split("\n") if line.strip()]
        return captions[:3] if len(captions) >= 3 else captions

    def generate(self, image_path: str, style: str = "General") -> dict:
        base_caption = self.captioner.generate_caption(image_path)
        context_docs = self.retriever.retrieve(base_caption, k=3, style=style)
        context_text = "\n".join(context_docs)
        final_captions = self.style_transform(base_caption, context_text, style)

        return {
            "base_caption": base_caption,
            "retrieved_context": context_docs,
            "final_captions": final_captions,
        }