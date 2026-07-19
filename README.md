# AI Research Assistant

An AI-powered PDF Question Answering application built using Python, Streamlit, LangChain, FAISS, and Groq Llama 3.3.

## Features

- Upload PDF documents
- Automatic text extraction
- Document chunking
- Semantic search using FAISS
- Retrieval-Augmented Generation (RAG)
- Natural language question answering
- Interactive Streamlit interface
- Session-based chat history

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Groq API (Llama 3.3 70B)
- PyPDF
- python-dotenv

## Project Workflow

1. Upload a PDF
2. Extract text from the document
3. Split text into chunks
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant chunks
7. Send retrieved context to Groq LLM
8. Generate context-aware answers

## Installation

Clone the repository

```bash
git clone https://github.com/yagnasreekamireddy/ai-research-assistant.git
