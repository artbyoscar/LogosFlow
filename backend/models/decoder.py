from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity

class Decoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def decode_embedding(self, embedding, corpus, corpus_embeddings, k=3):
        # Ensure embedding is a tensor and add a batch dimension
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.from_numpy(embedding).float()
        emb_normalized = embedding.unsqueeze(0) / embedding.norm()

        # Normalize corpus_embeddings
        corpus_embeddings_normalized = corpus_embeddings / torch.norm(corpus_embeddings, dim=1, keepdim=True)

        # Compute similarities
        similarities = cosine_similarity(emb_normalized, corpus_embeddings_normalized)

        # Get top k indices
        top_k_indices = similarities.topk(k).indices
        # Randomly select one of the top k indices
        selected_idx = top_k_indices[torch.randint(0, k, (1,))].item()
        selected_sentence = corpus[selected_idx]
        return selected_sentence, selected_idx