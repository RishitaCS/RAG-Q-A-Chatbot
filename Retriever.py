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
