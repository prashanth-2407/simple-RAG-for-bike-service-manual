import streamlit as st

# 🚨 Must be the first Streamlit command
st.set_page_config(page_title="Bike Manual Q&A", layout="wide")

import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import PyPDF2

# ------------------ Load Embedding Model ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ------------------ PDF Text Extraction ------------------
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """Extract text from a PDF file and split it into chunks."""
    text_chunks = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return text_chunks

# ------------------ Build FAISS Index ------------------
@st.cache_resource
def build_faiss_index(docs: List[str]):
    embeddings = embedding_model.encode(docs)
    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_docs(query: str, manuals: List[str], index, top_k: int = 2) -> List[str]:
    """Retrieve top-k relevant documents for a given query."""
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return [manuals[i] for i in indices[0]]

def generate_answer(query: str, manuals: List[str], index) -> str:
    """Retrieve relevant context and use Ollama to generate an answer."""
    instruction = (
        "As a bike service expert working in a bike service center, "
        "your role is to provide accurate answers to rider questions. "
        "Data in the context part is your knowledge about the bike. "
        "Stick to the context, but if data is missing, use your own knowledge."
    )
    retrieved_docs = retrieve_docs(query, manuals, index)
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nInstruction: {instruction}\n\nQuestion: {query}\nAnswer:"

    response = ollama.chat(model='wizardlm2', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ------------------ Streamlit UI ------------------
st.title("🏍️ Bike Troubleshooting Assistant")

# File uploader
pdf_file = st.file_uploader("Upload a troubleshooting manual (PDF)", type=["pdf"])

if pdf_file:
    with st.spinner("Processing PDF..."):
        # Save file temporarily
        pdf_path = f"uploaded_{pdf_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        # Extract & index
        manuals = extract_text_from_pdf(pdf_path)
        index, _ = build_faiss_index(manuals)

    st.success("PDF processed and indexed successfully ✅")

    # Chat UI
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query, manuals, index)

        st.markdown("### 📖 Answer")
        st.write(answer)

        # Optional: Show retrieved docs
        with st.expander("🔍 Retrieved Context"):
            for doc in retrieve_docs(user_query, manuals, index):
                st.markdown(doc)
else:
    st.info("📂 Please upload a troubleshooting manual PDF to begin.")
