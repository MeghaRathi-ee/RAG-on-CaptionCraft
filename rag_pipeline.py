from models.caption_model import CaptionGenerator
from retriever import Retriever


class RAGCaptionCraft:
    def __init__(self):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()

    def style_transform(
        self,
        caption,
        retrieved_examples,
        style: str | None = None,
    ):
        """
        Transform base caption into Instagram style caption
        using retrieved caption examples (few-shot RAG).
        """

        # Join retrieved examples into formatted block
        examples = "\n".join(f"- {ex}" for ex in retrieved_examples if ex)

        style_hint = f" in a {style} vibe" if style else ""

        prompt = f"""
You are an assistant that writes short, aesthetic Instagram captions{style_hint}.

Here are some example captions:
{examples}

Now write a very short, cute Gen-Z style Instagram caption with emojis
for this image description:
\"\"\"{caption}\"\"\"

Caption:
"""

        styled_caption = self.captioner.generate_text(prompt)

        return styled_caption.strip()

    def generate(
        self,
        image_path,
        mode: str = "full_rag",
        style: str | None = None,
        username: str | None = None,
    ):
        """
        Full RAG pipeline:
        1. Generate base caption from image (BLIP)
        2. Retrieve similar Instagram captions (ChromaDB)
        3. Generate styled caption using FLAN-T5
        """

        # Step 1: Generate image description (shared for all modes)
        base_caption = self.captioner.generate_caption(image_path)

        # Base-only mode (no style rewrite)
        if mode == "base":
            return {
                "mode": "base",
                "base_caption": base_caption,
                "retrieved_context": [],
                "final_caption": base_caption,
                "style_only_caption": None,
                "hashtags": "",
            }

        # Optional RAG retrieval
        retrieved_examples = []
        if mode in ("full_rag", "compare"):
            retrieved_examples = self.retriever.retrieve(
                base_caption,
                k=3,
                style=style,
                username=username,
            )

        # Style-only (no retrieved examples)
        style_only_caption = self.style_transform(
            base_caption,
            [],
            style=style,
        )

        # Full RAG caption uses retrieved examples
        full_rag_caption = self.style_transform(
            base_caption,
            retrieved_examples,
            style=style,
        )

        # Hashtag suggestion based on base caption and style
        hashtags = self.retriever.retrieve_hashtags(
            base_caption,
            style=style,
        )

        if mode == "style_only":
            final = style_only_caption
        elif mode == "full_rag":
            final = full_rag_caption
        else:  # "compare" returns all variants but uses full RAG as primary
            final = full_rag_caption

        return {
            "mode": mode,
            "base_caption": base_caption,
            "retrieved_context": retrieved_examples,
            "style_only_caption": style_only_caption,
            "full_rag_caption": full_rag_caption,
            "final_caption": final,
            "hashtags": hashtags,
        }

    def add_user_caption(self, username: str, caption: str) -> None:
        self.retriever.add_user_caption(username, caption)
