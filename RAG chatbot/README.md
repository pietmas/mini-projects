# RAG Chatbot

## Table of Contents
1. [Project Description](#project-description)
2. [Features](#features)
3. [Project Structure](#project-structure)

---

## Project Description

This repository contains a **retrieval-augmented generation (RAG)** chatbot built with:
- **LangChain** for pipeline orchestration and prompt/chain management.
- **FAISS** for vector similarity search.
- **Hugging Face Transformers** for the language model.

RAG allows the chatbot to:
1. Retrieve **relevant** snippets from your custom documents.
2. Use an **LLM** to generate a final answer with context from those snippets.

---

## Features

- **Document Preprocessing**  
  - Load `.txt` files from a `data/` folder.  
  - Chunk the documents (e.g., into 1,000-character sections) for better retrieval.

- **Vector Store with FAISS**  
  - Embed document chunks using a pretrained embedding model (`sentence-transformers/all-MiniLM-L6-v2`).  
  - Store embeddings in a FAISS index for fast approximate nearest neighbor (ANN) searches.

- **Open-Source LLM Integration**  
  - Use a Hugging Face model, `tiiuae/falcon-7b-instruct`, in a text-generation pipeline.  
  - Leverage GPU acceleration if available.

- **Easy RetrievalQA**  
  - Build a LangChain `RetrievalQA` chain that automatically performs similarity search and “stuffs” the relevant chunks into the LLM prompt.

- **Interactive Chat**  
  - Simple command-line interface (CLI) for asking questions.  
  - The system retrieves the best-matching context from your dataset before generating an answer.

---

## Project Structure

- **`data/`**: Directory containing your custom `.txt` documents.
    - **`docs1.txt`**: About the scout movement.
    - **`docs2.txt`**: About cycling.
- **`main.py`**: Main script to load documents, build the vector store, set up the model, and start the Q&A loop.  
- **`requirements.txt`**: List of Python dependencies.  
- **`README.md`**: This file, describing the project in detail.
