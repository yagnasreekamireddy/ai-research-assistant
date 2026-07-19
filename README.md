# 🤖 AI Research Assistant

A GenAI-powered document assistant that helps users interact with PDF files using natural language questions.

Built as a learning project to explore how **Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Vector Databases** can be combined to create useful AI applications.

## 🌟 Project Overview

Instead of manually searching through lengthy documents, this application allows users to upload a PDF and ask questions directly. The system understands the document, finds relevant information, and generates meaningful answers using AI.

The project uses a RAG pipeline where document content is converted into embeddings, stored in a vector database, and retrieved whenever a user asks a question.

## 🚀 Features

✅ Upload and analyze PDF documents  
✅ Ask questions using natural language  
✅ ChatGPT-like conversational interface  
✅ Semantic document search  
✅ Context-based AI responses  
✅ Maintains conversation history  
✅ Supports different types of documents

## 🔧 Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq Llama 3.3
- PyPDF

## ⚙️ How It Works

1. Upload a PDF document
2. Extract text from the document
3. Split content into smaller chunks
4. Convert chunks into embeddings
5. Store embeddings using FAISS
6. Retrieve relevant information for user queries
7. Generate answers using Llama LLM

## 🎯 Use Cases

- Resume analysis
- Research paper exploration
- Academic notes assistant
- Business document analysis
- Report understanding

## 📌 Key Learnings

Through this project, I explored:

- Building RAG applications
- Working with LLM APIs
- Vector similarity search
- Document processing pipelines
- Developing AI applications using Streamlit

## 🔮 Future Enhancements

- Multiple PDF support
- Source citations
- Cloud deployment
- Voice interaction
- Improved retrieval techniques
