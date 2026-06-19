import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv(".env")

api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Research Assistant")

st.title("📄 AI Research Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    st.session_state.chunks = splitter.split_text(text)

    st.success(
        f"PDF Loaded Successfully! "
        f"{len(st.session_state.chunks)} chunks created."
    )

question = st.text_input(
    "Ask a question about the PDF"
)

if question and st.session_state.chunks:

    query_words = question.lower().split()

    relevant_chunks = []

    for chunk in st.session_state.chunks:

        chunk_lower = chunk.lower()

        if any(
            word in chunk_lower
            for word in query_words
        ):
            relevant_chunks.append(chunk)

    context = "\n\n".join(
        relevant_chunks[:3]
    )

    if not context:
        context = "\n\n".join(
            st.session_state.chunks[:3]
        )

    response = llm.invoke(
        f"""
        Answer using the PDF context.

        Context:
        {context}

        Question:
        {question}
        """
    )

    answer = response.content

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

if st.button("Clear Chat"):
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:

    st.markdown(
        f"**You:** {chat['question']}"
    )

    st.info(
        chat["answer"]
    )