# AI Research Assistant

AI-powered PDF Question Answering application built using Streamlit, LangChain, and Groq Llama models.

## Features

* Upload PDF documents
* Extract and process document content
* Document chunking for efficient context retrieval
* Natural language question answering
* Interactive chat interface
* Session-based conversation history

## Tech Stack

* Python
* Streamlit
* LangChain
* Groq API (Llama 3.3 70B)
* PyPDF
* python-dotenv

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

## Future Improvements

* FAISS vector search
* Source citations
* Downloadable chat history
* Multi-document support
* Conversation memory

```
```
