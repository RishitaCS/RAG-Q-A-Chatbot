# 🧠 RAG-Powered Q&A Chatbot

A smart chatbot powered by **Retrieval-Augmented Generation (RAG)** using Hugging Face models and built with Streamlit for a clean, interactive user interface.

This chatbot reads custom documents, retrieves relevant content, and generates human-like answers using generative AI — combining the best of search and synthesis.

---

## 🚀 Features

- 📄 **Document Uploading** – Add your own `.txt` files
- 🔍 **Semantic Search** – Uses embeddings to retrieve the most relevant context
- 💬 **LLM-Powered Answers** – Uses `google/flan-t5-base` to generate coherent, factual answers
- 🖥️ **Streamlit UI** – Clean, user-friendly interface with chat history
- 🧠 **Memory-Powered** – Maintains a chat history per session for continuity
- ⚙️ **Lightweight & Fast** – Designed to run on Colab or CPU environments

---

## 🌐 Live Demo

👉 [Click here to try the chatbot app](https://rag-q-a-chatbot-exhgbwzj9wc9k58wxmjqed.streamlit.app/)

> Note: It may take a few seconds to load as it initializes the model on CPU.

## 📁 Folder Structure
RAG-Q-A-Chatbot/
│
├── app.py                  # 🔷 Main Streamlit app
├── Retriever.py            # 🔎 DocumentRetriever class
├── Generator.py            # 🧠 AnswerGenerator class
├── docs.txt                # 📄 Input documents
├── requirements.txt        # 📦 Dependencies for deployment
└── README.md               # 📘 Project documentation

**📚 How It Works**
1) Upload documents (default docs.txt)
2) Ask a question using the chat input
3) The system:
- Retrieves relevant context using all-MiniLM-L6-v2 embeddings
- Feeds context + question to google/flan-t5-base
- Returns an AI-generated answer
