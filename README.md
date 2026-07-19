# 🤖 AI Research Assistant

A GenAI-powered document assistant that helps users interact with PDF files using natural language questions.

Built as a learning project to explore how **Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Vector Databases** can be combined to build practical AI applications.

## 🌟 Project Overview

Reading and finding information from lengthy documents can be time-consuming. This application makes document analysis easier by allowing users to upload a PDF and ask questions directly.

The system understands the uploaded document, retrieves relevant information based on user queries, and generates meaningful answers using AI.

The project follows a **Retrieval-Augmented Generation (RAG)** approach where document content is converted into embeddings, stored in a vector database, and retrieved whenever a user asks a question.

## 🚀 Features

✅ Upload and analyze PDF documents  
✅ Ask questions using natural language  
✅ ChatGPT-like conversational interface  
✅ Semantic document search  
✅ Context-aware AI responses  
✅ Maintains conversation history  
✅ Supports different types of documents  

## 🔧 Technologies Used

- Python
- Streamlit
- LangChain
- FAISS Vector Database
- HuggingFace Embeddings
- Groq Llama 3.3
- PyPDF

## ⚙️ How It Works

```
Upload PDF Document
        ↓
Extract Text Content
        ↓
Split Text into Chunks
        ↓
Generate Embeddings
        ↓
Store in FAISS Vector Database
        ↓
Retrieve Relevant Information
        ↓
Generate Response using Llama LLM
```

## 💻 Installation

### Clone the repository

```bash
git clone https://github.com/yagnasreekamireddy/ai-research-assistant.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### Run the application

```bash
streamlit run app.py
```

## 🎯 Use Cases

- Resume analysis
- Research paper exploration
- Academic notes assistant
- Business document analysis
- Report understanding
- Knowledge extraction from documents

## 📌 Key Learnings

Through this project, I explored:

- Building Retrieval-Augmented Generation (RAG) applications
- Working with Large Language Model APIs
- Vector similarity search
- Document processing pipelines
- Embedding-based retrieval
- Developing AI applications using Streamlit

## 🔮 Future Enhancements

- Multiple PDF support
- Source citations with page references
- Cloud deployment
- Voice-based interaction
- Improved retrieval techniques

## 👩‍💻 Author

**Yagnasree Kamireddy**

B.Tech Computer Science Engineering (Data Science)
