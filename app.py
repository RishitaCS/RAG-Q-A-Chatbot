# -*- coding: utf-8 -*-


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
