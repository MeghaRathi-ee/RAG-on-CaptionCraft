import csv
from datetime import datetime, timezone
import os
import tempfile

import streamlit as st
from PIL import Image

from rag_pipeline import RAGCaptionCraft


STYLE_OPTIONS = [
    ("auto", "Auto-detect"),
    ("coffee", "Coffee / cozy"),
    ("sunset", "Sunset / golden hour"),
    ("selfie", "Selfie / outfit"),
    ("food", "Food / dessert"),
    ("travel", "Travel / vacation"),
    ("work", "Work / study"),
    ("selfcare", "Self-care / soft life"),
    ("growth", "Growth / new chapter"),
]


RATINGS_PATH = "data/ratings.csv"


# Page config
st.set_page_config(
    page_title="CaptionCraft RAG",
    page_icon="📸",
    layout="wide",
)

# Simple custom styling for a more professional look
st.markdown(
    """
    <style>
    body {
        background-color: #0f172a;
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1100px;
    }
    .caption-card {
        background: #020617;
        border-radius: 1rem;
        padding: 1.25rem 1.5rem;
        border: 1px solid #1f2937;
    }
    .caption-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .caption-text {
        font-size: 1.05rem;
        color: #e5e7eb;
    }
    .hashtag-box code {
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom: 0.2rem;'>📸 CaptionCraft</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9ca3af; font-size:0.95rem; margin-bottom:1.5rem;'>"
    "Professional-grade, style-aware Instagram captions powered by Retrieval-Augmented Generation."
    "</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag():
    return RAGCaptionCraft()


rag = load_rag()

# User & style controls
with st.sidebar:
    st.header("Options")
    username = st.text_input("Your name (for personal style)", value="")

    style_key = st.selectbox(
        "Choose a style (optional)",
        options=[k for k, _ in STYLE_OPTIONS],
        format_func=lambda k: dict(STYLE_OPTIONS)[k],
        index=0,
    )
    style = None if style_key == "auto" else style_key

    mode = st.radio(
        "Generation mode",
        options=["full_rag", "compare"],
        format_func=lambda m: "Full RAG only" if m == "full_rag" else "Compare modes",
    )


col_left, col_right = st.columns([1.1, 1.3])

with col_left:
    st.markdown("#### 1. Upload image")
    uploaded_file = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Preview",
            use_container_width=True,
        )

with col_right:
    st.markdown("#### 2. Generate caption")

    if uploaded_file is None:
        st.info("Upload an image on the left to get started.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        if st.button("Generate Caption 🚀", use_container_width=True):
            with st.spinner("Analyzing image and retrieving style examples..."):
                result = rag.generate(
                    temp_path,
                    mode=mode,
                    style=style,
                    username=username or None,
                )

            os.remove(temp_path)

            st.markdown("#### 3. Results")

            # Base caption card
            st.markdown(
                "<div class='caption-card'>"
                "<div class='caption-label'>Base caption</div>"
                f"<div class='caption-text'>{result['base_caption']}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            if result.get("retrieved_context"):
                st.markdown("<br/><div class='caption-label'>Retrieved style examples</div>", unsafe_allow_html=True)
                for example in result["retrieved_context"]:
                    st.markdown(f"- {example}")

            if mode == "compare":
                st.markdown("<br/><div class='caption-label'>Mode comparison</div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Base only**")
                    st.write(result["base_caption"])

                with c2:
                    st.markdown("**Style prompt (no RAG)**")
                    st.write(result.get("style_only_caption") or "")

                with c3:
                    st.markdown("**Full RAG**")
                    st.write(result.get("full_rag_caption") or "")
            else:
                st.markdown("<br/><div class='caption-label'>Final Instagram caption</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='caption-card'>"
                    f"<div class='caption-text'>{result['final_caption']}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            if result.get("hashtags"):
                st.markdown("<br/><div class='caption-label'>Suggested hashtags</div>", unsafe_allow_html=True)
                st.code(result["hashtags"])

            # Personalization: allow saving final caption to user style
            if username and st.button("Save final caption to my style"):
                rag.add_user_caption(username, result["final_caption"])
                st.success("Saved to your personal style memory.")

            # Simple evaluation form
            if mode == "compare":
                st.markdown("<br/><div class='caption-label'>Rate these captions</div>", unsafe_allow_html=True)
                with st.form("ratings"):
                    base_rel = st.slider(
                        "Base caption – relevance", 1, 5, 3
                    )
                    style_rel = st.slider(
                        "Style-only caption – relevance", 1, 5, 3
                    )
                    rag_rel = st.slider(
                        "Full RAG caption – relevance", 1, 5, 4
                    )

                    rag_style = st.slider(
                        "Full RAG caption – Instagram style / vibe", 1, 5, 4
                    )

                    submitted = st.form_submit_button("Save ratings")

                    if submitted:
                        os.makedirs(os.path.dirname(RATINGS_PATH), exist_ok=True)
                        file_exists = os.path.exists(RATINGS_PATH)

                        with open(RATINGS_PATH, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(
                                    [
                                        "timestamp",
                                        "username",
                                        "image_name",
                                        "style_choice",
                                        "base_caption",
                                        "style_only_caption",
                                        "full_rag_caption",
                                        "base_rel",
                                        "style_rel",
                                        "rag_rel",
                                        "rag_style",
                                    ]
                                )

                            writer.writerow(
                                [
                                    datetime.now(timezone.utc).isoformat(),
                                    username,
                                    getattr(uploaded_file, "name", ""),
                                    style or "auto",
                                    result["base_caption"],
                                    result.get("style_only_caption") or "",
                                    result.get("full_rag_caption") or result["final_caption"],
                                    base_rel,
                                    style_rel,
                                    rag_rel,
                                    rag_style,
                                ]
                            )

                        st.success("Ratings saved. Thank you!")
