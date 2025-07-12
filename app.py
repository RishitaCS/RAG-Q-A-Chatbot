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

retriever = DocumentRetriever("Doc.txt")
generator = AnswerGenerator()

while True:
    query = input("\n Ask a question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    context = "\n".join(retriever.retrieve(query))
    print("\n Retrieved Context:\n", context)
    answer = generator.generate(query, context)
    print("\n Answer:\n", answer)
