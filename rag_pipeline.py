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

    def _call_groq(self, prompt: str, max_tokens: int = 300) -> str:
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Groq error: {str(e)}"

    def style_transform(self, caption: str, context: str, style: str) -> list[str]:

        if style == "Story":
            prompt = f"""Write 3 separate Instagram story captions for this image: "{caption}"

Rules:
- Write exactly 3 captions, each completely independent
- Each caption must be 8 lines long
- Line 1-2: Set the scene vividly - time, place, feeling
- Line 3-4: Build the emotional core of the story
- Line 5-6: Add a personal reflection or turning point
- Line 7: A powerful closing sentence
- Line 8: 4-5 relevant hashtags
- Write in first person, raw and honest
- Use emojis naturally within the text
- Separate each caption with this exact separator: ---
- No numbering, no labels
- Each caption should tell a completely different story about the same image"""

            raw = self._call_groq(prompt, max_tokens=1024)
            captions = [block.strip() for block in raw.split("---") if block.strip()]

        else:
            prompt = f"""Write 3 Instagram captions for this image: "{caption}"
Style: {style}
Each caption on its own line. Include emojis and 3 hashtags. No numbering."""

            raw = self._call_groq(prompt, max_tokens=300)
            captions = [line.strip() for line in raw.split("\n") if line.strip()]

        return captions[:3] if len(captions) >= 3 else captions

    def generate(self, image_path: str, style: str = "General") -> dict:
        # Step 1: Vision model
        base_caption = self.captioner.generate_caption(image_path)

        # Step 2: Retrieve top-3 relevant style docs
        context_docs = self.retriever.retrieve(base_caption, k=3, style=style)
        context_text = "\n".join(context_docs)

        # Step 3: LLM generation
        final_captions = self.style_transform(base_caption, context_text, style)

        return {
            "base_caption": base_caption,
            "retrieved_context": context_docs,
            "final_captions": final_captions,
        }