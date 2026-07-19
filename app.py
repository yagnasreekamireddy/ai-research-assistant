import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

api_key = os.getenv("M")

st.set_page_config(page_title="AI Research Assistant")

st.title("📄 AI Research Assistant")

st.write("Upload a PDF and ask questions about its contents.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

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

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.session_state.vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    st.success(
        f"PDF Loaded Successfully! {len(chunks)} chunks created and indexed."
    )

# Display previous conversation
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

question = st.chat_input("Ask a question about the PDF")
if question and st.session_state.vector_store is not None:

    with st.chat_message("user"):
        st.markdown(question)

    docs = st.session_state.vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(
    f"""
You are an AI Research Assistant.

Answer the user's question using only the document context.

Context:
{context}

User Question:
{question}
Instructions:
- Answer in a structured format with headings and bullet points.
- Include all relevant information from the document.
- Do not omit any projects, skills, or important details.
- If the information is not available in the document, clearly say:
  "The uploaded document does not contain this information."
"""
)

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )
