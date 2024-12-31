from sentence_transformers import SentenceTransformer, util
import torch

class Decoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def decode_embedding(self, embedding, corpus, corpus_embeddings=None):
        if corpus_embeddings is None:
            corpus_embeddings = self.model.encode(corpus)
        cos_scores = util.cos_sim(embedding, corpus_embeddings)[0]
        top_result_index = cos_scores.argmax()
        return corpus[top_result_index], corpus_embeddings