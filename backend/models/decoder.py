import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import random

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(Decoder, self).__init__()
        self.model = SentenceTransformer(model_name)
        
        # Initialize Transformer Decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=hidden_size, dropout=0.1),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # You may need to adjust the output dimension

    def decode_embedding(self, embedding, corpus, corpus_embeddings, k=5):
        # Ensure embedding is a tensor and has the correct shape
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension if missing

        # Normalize the embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Ensure corpus_embeddings is a tensor and normalize it
        if not isinstance(corpus_embeddings, torch.Tensor):
            corpus_embeddings = torch.tensor(corpus_embeddings)
        corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(embedding, corpus_embeddings)
        similarities = torch.clamp(similarities, min=1e-6)  # Add a small value to prevent near-zero values
        
        # Check for NaN values in similarities
        if torch.isnan(similarities).any():
            print("Warning: NaN values found in similarities.")
            similarities = torch.nan_to_num(similarities, nan=0.0)  # Replace NaN with 0.0
        
        print("Similarities:", similarities)
        print("Embedding shape:", embedding.shape)
        print("Corpus embeddings shape:", corpus_embeddings.shape)
        
        # Get top k indices and similarities
        k = min(k, similarities.numel())  # Use at most the number of available similarities
        top_k_similarities, top_k_indices = similarities.topk(k)
        print("top_k_indices:", top_k_indices)
        print("Top k similarities shape:", top_k_similarities.shape)
        
        # Check if top_k_indices is empty
        if top_k_indices.numel() == 0:
            print("Warning: No similar sentences found. Returning defaults.")
            return "", -1, 0.0  # Return default values
        
        # Simplify top_k_indices to 1D if it's 2D with a single batch
        if top_k_indices.dim() == 2 and top_k_indices.size(0) == 1:
            top_k_indices = top_k_indices.squeeze(0)
            top_k_similarities = top_k_similarities.squeeze(0)
        
        # Select a random index from the top k
        random_index = random.randint(0, k - 1)
        selected_index = top_k_indices[random_index].item()
        selected_similarity = top_k_similarities[random_index].item()
        
        print("Selected index:", selected_index)
        print("Selected similarity:", selected_similarity)
        
        # Retrieve the corresponding sentence from the corpus
        decoded_sentence = corpus[selected_index]
        
        return decoded_sentence, selected_index, selected_similarity