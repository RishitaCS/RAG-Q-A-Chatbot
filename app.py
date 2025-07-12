# -*- coding: utf-8 -*-

# app.py

import streamlit as st
from Retriever import DocumentRetriever
from Generator import AnswerGenerator
import tempfile
import os
import PyPDF2

# Streamlit page setup
st.set_page_config(page_title="RAG Q&A Chatbot", layout="centered")
st.title("RAG-Powered Q&A Chatbot")
st.markdown("Upload documents (TXT or PDF) and ask questions based on them.")

# Upload area
uploaded_files = st.file_uploader("Upload one or more documents", type=["txt", "pdf"], accept_multiple_files=True)

# Load documents and create retriever dynamically
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("Ask a question based on the uploaded documents:")

# Handle query
if query:
    with st.spinner("Thinking..."):
        context = "\n".join(retriever.retrieve(query))
        answer = generator.generate(query, context)
        st.session_state.chat_history.append({
            "user": query,
            "context": context,
            "bot": answer
        })

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(f"**Answer:** {chat['bot']}")
        with st.expander("Show Retrieved Context"):
            st.write(chat["context"])
