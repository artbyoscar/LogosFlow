from sentence_transformers import SentenceTransformer

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_sentences(self, sentences):
        return self.model.encode(sentences)