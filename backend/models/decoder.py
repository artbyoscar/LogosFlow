import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
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
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Adjust the output dimension if necessary

    def decode_embedding(self, embedding, corpus, corpus_embeddings, k=5):
        """
        Decodes the given embedding(s) by finding the top-k similar embeddings in the corpus.

        Args:
            embedding (torch.Tensor): Tensor of shape [batch_size, embedding_dim].
            corpus (List[str]): List of sentences or data points in the corpus.
            corpus_embeddings (torch.Tensor): Tensor of shape [corpus_size, embedding_dim].
            k (int): Number of top similar items to retrieve.

        Returns:
            List[Tuple[str, int, float]]: A list of tuples containing the decoded sentence,
                                         its index in the corpus, and the similarity score.
        """
        # Debug Prints
        print("Shape of embedding (in decoder):", embedding.shape) 
        print("Shape of corpus_embeddings (in decoder):", corpus_embeddings.shape) 

        # Ensure embedding is a tensor and has the correct shape
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # If embedding is 3D (e.g., [batch_size, seq_len, embedding_dim]), select the last sequence step
        if embedding.dim() == 3:
            embedding = embedding[:, -1, :]  # [batch_size, embedding_dim]
        
        # Add batch dimension if missing (assuming single embedding)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # [1, embedding_dim]
        
        # Normalize the embedding
        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)  # [batch_size, embedding_dim]
        
        # Ensure corpus_embeddings is a tensor and normalize it
        if not isinstance(corpus_embeddings, torch.Tensor):
            corpus_embeddings = torch.tensor(corpus_embeddings, dtype=torch.float32)
        corpus_norm = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)  # [corpus_size, embedding_dim]
        
        # Compute cosine similarity using matrix multiplication
        # Resulting shape: [batch_size, corpus_size]
        similarities = torch.mm(embedding_norm, corpus_norm.t())
        
        # Clamp similarities to prevent extremely small values (optional, based on your use case)
        similarities = torch.clamp(similarities, min=1e-6)
        
        # Check for NaN values in similarities
        if torch.isnan(similarities).any():
            print("Warning: NaN values found in similarities.")
            similarities = torch.nan_to_num(similarities, nan=0.0)  # Replace NaN with 0.0
        
        print("Similarities:", similarities)  # Debug Print
        
        # Adjust k to not exceed the number of corpus items
        k = min(k, similarities.size(1))  # similarities.size(1) == corpus_size
        
        # Get top k similarities and their indices for each embedding in the batch
        top_k_similarities, top_k_indices = similarities.topk(k, dim=1, largest=True, sorted=True)  # [batch_size, k]
        
        print("Top k indices:", top_k_indices)  # Debug Print
        print("Top k similarities:", top_k_similarities)  # Debug Print
        
        # Initialize list to hold decoded results
        decoded_results = []
        
        # Iterate over each item in the batch
        batch_size = embedding.size(0)
        for i in range(batch_size):
            # Check if top_k_indices is empty for this batch item
            if top_k_indices[i].numel() == 0:
                print(f"Warning: No similar sentences found for batch item {i}. Returning defaults.")
                decoded_results.append(("", -1, 0.0))  # Append default values
                continue
            
            # Select a random index from the top k for this batch item
            random_index = random.randint(0, k - 1)
            selected_index = top_k_indices[i][random_index].item()
            selected_similarity = top_k_similarities[i][random_index].item()
            
            print(f"Batch {i} - Selected index:", selected_index)  # Debug Print
            print(f"Batch {i} - Selected similarity:", selected_similarity)  # Debug Print
            
            # Retrieve the corresponding sentence from the corpus
            decoded_sentence = corpus[selected_index]
            
            # Append the result as a tuple
            decoded_results.append((decoded_sentence, selected_index, selected_similarity))
        
        return decoded_results
