# 🖼️ CaptionCraft-RAG

### Context-Aware Image Captioning using Retrieval-Augmented Generation (RAG)

CaptionCraft-RAG is an intelligent image captioning system enhanced with **Retrieval-Augmented Generation (RAG)**.
It generates stylish, context-aware captions by combining:

* 🧠 Vision Model (BLIP)
* 🔎 Vector Retrieval (ChromaDB)
* 📚 Text Embeddings (Sentence-Transformers)
* 🎨 Interactive Web UI (Streamlit)

---

## 🚀 Features

* Upload an image
* Generate base caption using BLIP
* Retrieve contextual style knowledge
* Produce styled Instagram-like captions
* Interactive Streamlit web interface
* CPU-friendly implementation

---

## 🏗️ System Architecture

```
          Input Image
               ↓
      BLIP Image Captioning
               ↓
         Base Caption
               ↓
      SentenceTransformer
               ↓
         ChromaDB Retrieval
               ↓
      Style Transformation
               ↓
        Final RAG Caption
```

---

## 🛠️ Tech Stack

| Component        | Technology       |
| ---------------- | ---------------- |
| Image Captioning | Salesforce BLIP  |
| Embeddings       | all-MiniLM-L6-v2 |
| Vector Database  | ChromaDB         |
| Backend          | Python           |
| UI               | Streamlit        |

---

## 📂 Project Structure

```
caption_craft/
│
├── data/
│   ├── images/
│   └── knowledge/
│
├── models/
│   ├── caption_model.py
│   └── embedding_model.py
│
├── vector_store/
│
├── build_index.py
├── retriever.py
├── rag_pipeline.py
├── app.py
├── streamlit_app.py
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd caption_craft
```

### 2️⃣ Create Conda Environment

```bash
conda create -n captioncraft python=3.10
conda activate captioncraft
```

### 3️⃣ Install Dependencies

```bash
pip install torch torchvision transformers
pip install sentence-transformers chromadb
pip install streamlit pillow
```

---

## 🔧 Build Vector Index

Before running the application, build the ChromaDB knowledge index:

```bash
python build_index.py
```

Expected output:

```
✅ ChromaDB index created successfully
```

---

## ▶️ Run the Application

### CLI Version:

```bash
python app.py
```

### Streamlit Web Interface:

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧠 Example Output

**Base Caption:**

> a woman sitting on a couch with a cup of coffee

**Retrieved Context:**

> Instagram captions are short, casual, and emoji-friendly.

**Final RAG Caption:**

> coffee vibes ☕✨

---

## 💡 Why RAG?

Traditional image captioning models rely only on visual features.

CaptionCraft-RAG enhances generation by:

* Retrieving contextual style knowledge
* Augmenting caption generation with retrieved information
* Producing expressive and domain-aware captions

This demonstrates integration of **Computer Vision + NLP + Vector Databases** in a unified pipeline.

---

## 🔮 Future Improvements

* Multiple style selection (Funny, Formal, Travel, Food)
* LLM-based caption refinement
* Multilingual support
* Cloud deployment
* REST API integration

---

## 👩‍💻 Author

**Megha Rathi**
M.E. Artificial Intelligence & Machine Learning

---

## 📜 License

This project is developed for academic and educational purposes.

---

## ⭐ If You Like This Project

Feel free to star ⭐ the repository and contribute!

---
