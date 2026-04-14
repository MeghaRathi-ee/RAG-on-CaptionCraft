import ollama

from models.caption_model import CaptionGenerator
from retriever import Retriever


class RAGCaptionCraft:

    def __init__(self, model: str = "llama3"):
        self.captioner = CaptionGenerator()
        self.retriever = Retriever()
        self.model = model
        self._user_style_memory: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_full_rag_prompt(
        self,
        base_caption: str,
        context: list[str],
        style: str | None,
        username: str | None,
    ) -> str:
        context_block = "\n".join(f"- {c}" for c in context)
        style_line = f"Style theme: {style}\n" if style else ""
        user_line = ""
        if username and username in self._user_style_memory:
            examples = "\n".join(f"- {c}" for c in self._user_style_memory[username][-3:])
            user_line = f"Personal style examples from {username}:\n{examples}\n"

        return (
            f"You are a creative Instagram caption writer.\n\n"
            f"Image described as: \"{base_caption}\"\n\n"
            f"Retrieved style knowledge:\n{context_block}\n\n"
            f"{style_line}"
            f"{user_line}"
            f"Write ONE short, punchy Instagram caption (max 15 words). "
            f"Include 1-2 relevant emojis. Output only the caption, nothing else."
        )

    def _build_style_only_prompt(self, base_caption: str, style: str | None) -> str:
        style_line = f"Style theme: {style}\n" if style else ""
        return (
            f"You are a creative Instagram caption writer.\n\n"
            f"Image described as: \"{base_caption}\"\n\n"
            f"{style_line}"
            f"Write ONE short, punchy Instagram caption (max 15 words). "
            f"Include 1-2 relevant emojis. Output only the caption, nothing else."
        )

    def _build_hashtag_prompt(self, caption: str, style: str | None) -> str:
        style_line = f"Style theme: {style}. " if style else ""
        return (
            f"Generate 5-8 relevant Instagram hashtags for this caption: \"{caption}\". "
            f"{style_line}"
            f"Output only the hashtags separated by spaces, nothing else."
        )

    # ------------------------------------------------------------------ #
    #  LLM call                                                            #
    # ------------------------------------------------------------------ #

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_path: str,
        mode: str = "full_rag",
        style: str | None = None,
        username: str | None = None,
    ) -> dict:
        # Step 1: base caption from BLIP
        base_caption = self.captioner.generate_caption(image_path)

        # Step 2: retrieve context from ChromaDB
        retrieved_context = self.retriever.retrieve(base_caption, k=3)

        # Step 3: generate captions depending on mode
        full_rag_caption = self._chat(
            self._build_full_rag_prompt(base_caption, retrieved_context, style, username)
        )
        hashtags = self._chat(self._build_hashtag_prompt(full_rag_caption, style))

        result = {
            "base_caption": base_caption,
            "retrieved_context": retrieved_context,
            "final_caption": full_rag_caption,
            "full_rag_caption": full_rag_caption,
            "hashtags": hashtags,
            "style_only_caption": None,
        }

        if mode == "compare":
            result["style_only_caption"] = self._chat(
                self._build_style_only_prompt(base_caption, style)
            )

        return result

    def add_user_caption(self, username: str, caption: str) -> None:
        self._user_style_memory.setdefault(username, []).append(caption)