# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer, util
import torch

class DocumentRetriever:
    def __init__(self, docs_path: str):
        self.docs = self._load_docs(docs_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.docs, convert_to_tensor=True)

    def _load_docs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)
        return [self.docs[hit['corpus_id']] for hit in hits[0]]

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class AnswerGenerator:
    def __init__(self, model_name="google/flan-t5-base"):  # Smaller model for Colab
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, query, context):
        prompt = f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
        response = self.pipeline(prompt, max_new_tokens=200, do_sample=True)
        return response[0]['generated_text']

import streamlit as st
from Retriever import DocumentRetriever
from Generator import AnswerGenerator

# Initialize retriever and generator
retriever = DocumentRetriever("docs.txt")
generator = AnswerGenerator()

# Page setup
st.set_page_config(page_title="RAG Q&A Chatbot", layout="centered")
st.title("RAG-Powered Q&A Chatbot")
st.markdown("Ask any question based on the uploaded documents!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input area
with st.chat_input("Type your question...") as chat_input:
    query = st.session_state.get("user_input", "")
    if chat_input:
        st.session_state.user_input = chat_input

# Handle question & response
if st.session_state.get("user_input"):
    query = st.session_state.user_input
    with st.spinner("Thinking... "):
        context = "\n".join(retriever.retrieve(query))
        answer = generator.generate(query, context)

    # Save to chat history
    st.session_state.chat_history.append({
        "user": query,
        "context": context,
        "bot": answer
    })
    
    st.session_state.user_input = ""

# Display the full chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(f"**Answer:** {chat['bot']}")
        with st.expander("Show Retrieved Context"):
            st.write(chat["context"])
