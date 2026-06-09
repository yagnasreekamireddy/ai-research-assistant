from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

def load_pdf(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

chat_history = []
pdf_context = ""

print("=== AI Research Assistant ===")
print("Type 'load: filename.pdf' to load a PDF")
print("Type 'quit' to exit")
print("==============================\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    elif user_input.lower().startswith("load:"):
        file_path = user_input[5:].strip()
        try:
            pdf_context = load_pdf(file_path)
            print("\nAssistant: PDF loaded! Ask me anything about it.\n")
        except:
            print("\nAssistant: Could not load file.\n")
    else:
        messages = []
        if pdf_context:
            messages.append(SystemMessage(content=f"Answer using this document:\n\n{pdf_context[:4000]}"))
        messages += chat_history
        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))
        print(f"\nAssistant: {response.content}\n")