# AI Research Assistant

An AI-powered PDF research assistant built with Streamlit. Upload PDF documents and ask questions in natural language using Retrieval-Augmented Generation (RAG).

## Features

- Upload and preview PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate embeddings using HuggingFace
- Store and retrieve document vectors using FAISS
- Perform semantic similarity search
- Generate answers using Groq API and GPT-OSS 20B
- Context-aware question answering
- Session-based chat history

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- GPT-OSS 20B (`openai/gpt-oss-20b`)
- HuggingFace Embeddings
- FAISS
- PyPDF
- python-dotenv

## Installation

Clone the repository:

```bash
git clone https://github.com/yagnasreekamireddy/ai-research-assistant.git
cd ai-research-assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Replace the placeholder with your own Groq API key. Never upload your actual API key to GitHub.

## Run the Application

```bash
streamlit run app.py
```

## Project Structure

```text
ai-research-assistant/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## How It Works

1. Upload a PDF document.
2. Extract and split its text into chunks.
3. Generate embeddings for document chunks.
4. Store embeddings in a FAISS vector database.
5. Retrieve relevant content when a question is asked.
6. Generate an answer using retrieved context and the Groq API.
