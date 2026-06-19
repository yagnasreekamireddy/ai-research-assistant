# AI Research Assistant

A Streamlit-based AI assistant that allows users to upload PDF documents and ask questions using natural language.

## Features

* PDF upload and processing
* Document chunking using LangChain
* Context-based retrieval
* AI-powered question answering using Groq Llama models
* Interactive chat interface
* Session-based chat history

## Technologies Used

* Python
* Streamlit
* LangChain
* Groq API
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

Run:

```bash
streamlit run app.py
```

## Project Structure

```text
app.py
requirements.txt
README.md
.env
```
