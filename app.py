import streamlit as st
from retriever import DocumentRetriever
from generator import AnswerGenerator
import tempfile
import os
import PyPDF2

# Page config
st.set_page_config(page_title="RAG Q&A Chatbot", layout="centered")
st.title("RAG-Powered Q&A Chatbot")
st.markdown("📄 Upload documents (TXT or PDF) and ask questions based on them.")

# File upload
uploaded_files = st.file_uploader("Upload one or more documents", type=["txt", "pdf"], accept_multiple_files=True)

documents = []

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain":
            documents.append(uploaded_file.read().decode("utf-8"))
        elif uploaded_file.type == "application/pdf":
            documents.append(read_pdf(uploaded_file))

if documents:
    with st.spinner("Embedding uploaded documents..."):
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write("\n".join(documents))
            temp_doc_path = temp_file.name
        retriever = DocumentRetriever(temp_doc_path)
        generator = AnswerGenerator()
else:
    st.warning("Please upload at least one document to begin.")
    st.stop()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past chat
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
        with st.expander("🔍 Show Retrieved Context"):
            st.write(chat["context"])

# Input box (reappears after each message)
user_input = st.chat_input("Ask a question")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        context = "\n".join(retriever.retrieve(user_input))
        answer = generator.generate(user_input, context)

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("🔍 Show Retrieved Context"):
            st.write(context)

    st.session_state.chat_history.append({
        "user": user_input,
        "bot": answer,
        "context": context
    })
