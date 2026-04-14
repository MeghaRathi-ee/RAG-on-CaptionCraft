import streamlit as st
from PIL import Image
import tempfile
import os
from rag_pipeline import RAGCaptionCraft

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CaptionCraft RAG",
    page_icon="✨",
    layout="centered",
)

st.title("✨ CaptionCraft RAG")
st.markdown("**Upload an image → pick a style → get 3 Instagram-ready captions**")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar: style selection + tips
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    style = st.selectbox(
        "Caption Style",
        ["General", "Travel", "Food", "Fitness", "Aesthetic", "Funny", "Motivational", "Story"],
        help="This controls the tone and hashtags of your captions.",
    )

    st.divider()
    st.markdown("**Tips**")
    st.markdown(
        "- Run `ollama serve` before launching\n"
        "- Run `python build_index.py` once to build the vector index\n"
        "- Supports JPG, PNG, WEBP"
    )

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🚀 Generate Captions", type="primary", use_container_width=True):

        # Save to temp file so BLIP can load from path
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        ) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        with st.spinner("Running RAG pipeline... ⏳"):
            try:
                rag = RAGCaptionCraft()
                result = rag.generate(tmp_path, style=style)
            finally:
                os.unlink(tmp_path)

        # ---------------------------------------------------------------
        # Results
        # ---------------------------------------------------------------
        st.divider()

        with st.expander("🔍 Base Caption (from BLIP)", expanded=False):
            st.info(result["base_caption"])

        with st.expander("📚 Retrieved Style Context (from ChromaDB)", expanded=False):
            for i, doc in enumerate(result["retrieved_context"], 1):
                st.markdown(f"**Doc {i}:** {doc}")

        st.subheader(f"🎨 Generated Captions — *{style}* style")

        captions = result["final_captions"]
        if not captions:
            st.warning("No captions returned. Check if Ollama is running.")
        else:
            for i, caption in enumerate(captions, 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.text_area(
                        label=f"Option {i}",
                        value=caption,
                        height=300 if style == "Story" else 100,
                        key=f"caption_{i}",
                    )
                with col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    # text_area is already selectable/copyable — button for UX
                    if st.button("📋", key=f"copy_{i}", help="Select all text to copy"):
                        st.toast(f"Caption {i} ready to copy!")

        st.success("Done! Select any caption text above and copy it to Instagram.")

else:
    st.markdown(
        "<div style='text-align:center; color:grey; margin-top:3rem'>"
        "⬆️ Upload an image to get started"
        "</div>",
        unsafe_allow_html=True,
    )