# 🏍️ Bike Troubleshooting Assistant (RAG with FAISS + Ollama)

## 📖 About the Project
This project is a **Retrieval-Augmented Generation (RAG) pipeline** built around a bike troubleshooting manual.  
It takes a PDF manual as input, retrieves relevant sections using semantic search, and generates **context-aware answers** with an LLM (Ollama).  

- **Input**: Bike manual in PDF format  
- **Output**: Natural language answers to rider queries  
- **Pipeline Steps**:
  1. Extract text from the manual (**PyPDF2**)  
  2. Convert text into embeddings (**Sentence Transformers**)  
  3. Store & query embeddings using **FAISS**  
  4. Retrieve relevant chunks for a user query  
  5. Generate an answer with **Ollama** (`wizardlm2` model)  

---

## ⚙️ Tech Stack
- **Python**
- **PyPDF2** → PDF text extraction  
- **Sentence-Transformers** → embedding generation  
- **FAISS** → semantic search  
- **Ollama** → large language model inference  

---

## 🚴 Example Usage
```python
# Example query
query = "give some interesting facts about the bike?"
answer = generate_answer(query)
print("Answer:", answer)
