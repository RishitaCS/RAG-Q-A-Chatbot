from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class AnswerGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, query, context):
        prompt = f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
        response = self.pipeline(prompt, max_new_tokens=200, do_sample=True)
        return response[0]['generated_text']
