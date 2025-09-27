Bike Troubleshooting Assistant (RAG with FAISS + Ollama)

📖 About the Project

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built around a bike troubleshooting manual. It takes a PDF manual as input, retrieves relevant sections using semantic search, and generates context-aware answers with an LLM (Ollama).

Input: Bike manual in PDF format

Output: Natural language answers to rider queries

Approach:

Extracts text from the manual (PyPDF2).

Embeds documents using sentence-transformers.

Indexes embeddings in FAISS for semantic search.

Retrieves relevant chunks for a given query.

Generates answers with Ollama (wizardlm2 model).

⚙️ Tech Stack

Python

PyPDF2 (text extraction from PDF)

Sentence-Transformers (embedding generation)

FAISS (semantic search / nearest neighbor retrieval)

Ollama (LLM inference)

🚴 Use Case Example
Query: "give some interesting facts about the bike?"


✅ Pipeline retrieves relevant sections from the manual
✅ Ollama generates a natural response grounded in the context

📌 Why This Project?

Showcases practical application of RAG in a real-world domain (bike troubleshooting).

Demonstrates hybrid reasoning: knowledge from documents + model priors.

Scalable to other manuals (cars, electronics, appliances).
