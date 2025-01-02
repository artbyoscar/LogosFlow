# backend/models/generate.py

import torch
import numpy as np
import os
import sys
import gc
from torch.nn.functional import normalize
from pathlib import Path

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.model import SimplePolicyNetwork
from backend.models.decoder import Decoder
from backend.models.encoder import Encoder
from backend.utils.corpus_manager import CorpusManager

def apply_repetition_penalty(probabilities, generated_indices, penalty=1.2):
    """
    Applies a repetition penalty to the probabilities of already generated indices.

    Args:
        probabilities (torch.Tensor): The probability distribution over the vocabulary.
        generated_indices (List[int]): List of previously generated indices.
        penalty (float): The penalty factor to apply.

    Returns:
        torch.Tensor: The adjusted probability distribution.
    """
    if len(generated_indices) > 0:
        for idx in generated_indices[-5:]:
            if idx < probabilities.size(0):
                probabilities[idx] = probabilities[idx] / penalty
        probabilities = probabilities / probabilities.sum()
    return probabilities

def generate_sequence(model, start_embeddings, length, corpus, corpus_embeddings, device, k=15, temperature=1.5, top_p=0.9):
    """
    Generates a sequence of embeddings by iteratively sampling from the model's predictions.

    Args:
        model (nn.Module): The policy network model.
        start_embeddings (torch.Tensor): The initial embeddings to start generation.
        length (int): The number of embeddings to generate.
        corpus (List[str]): The corpus of sentences.
        corpus_embeddings (torch.Tensor): Precomputed embeddings for the corpus.
        device (torch.device): The device to run computations on.
        k (int): The number of top candidates to consider during sampling.
        temperature (float): The temperature parameter for scaling similarities.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        Tuple[List[torch.Tensor], List[int]]: Generated embeddings and their corresponding corpus indices.
    """
    model.eval()
    generated_embeddings = []
    closest_indices = []
    torch.set_num_threads(4)
    gc.collect()

    current_embeddings = start_embeddings.unsqueeze(0).to(device)  # [1, 1, embedding_dim]

    with torch.no_grad():
        for step in range(length):
            if current_embeddings.size(1) > 32:
                current_embeddings = current_embeddings[:, -32:, :]  # Truncate to last 32 embeddings

            next_embedding = model(current_embeddings)  # [1, 1, embedding_dim]
            next_embedding_normalized = torch.nn.functional.normalize(next_embedding.squeeze(0), dim=1)  # [1, embedding_dim]
            corpus_embeddings_normalized = torch.nn.functional.normalize(corpus_embeddings, dim=1)  # [corpus_size, embedding_dim]

            # Compute cosine similarities and apply temperature
            similarities = torch.matmul(next_embedding_normalized, corpus_embeddings_normalized.t()).squeeze(0)  # [corpus_size]
            scaled_similarities = similarities / temperature

            # Nucleus sampling
            probabilities = torch.nn.functional.softmax(scaled_similarities, dim=0)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask right to keep at least one token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            probabilities[indices_to_remove] = 0
            probabilities = probabilities / probabilities.sum()

            # Apply repetition penalty
            probabilities = apply_repetition_penalty(probabilities, closest_indices)

            # Sample from the filtered distribution
            top_k = min(k, corpus_embeddings.size(0))
            top_k_values, top_k_indices = probabilities.topk(top_k)
            top_k_probs = top_k_values / top_k_values.sum()
            selected_idx = top_k_indices[torch.multinomial(top_k_probs, 1)].item()

            closest_indices.append(selected_idx)
            generated_embeddings.append(next_embedding.squeeze(0).cpu())

            # Update current_embeddings with the new embedding
            next_embedding = next_embedding.to(device)  # Ensure it's on the correct device
            current_embeddings = torch.cat([current_embeddings, next_embedding], dim=1)  # [1, seq_len+1, embedding_dim]

            if step % 5 == 0:
                gc.collect()

    return generated_embeddings, closest_indices

def beam_search(model, start_embeddings, corpus_embeddings, device, beam_width=5, length=15, repetition_penalty=1.2):
    """
    Performs beam search to generate a sequence of embeddings.

    Args:
        model (nn.Module): The policy network model.
        start_embeddings (torch.Tensor): The initial embeddings to start generation.
        corpus_embeddings (torch.Tensor): Precomputed embeddings for the corpus.
        device (torch.device): The device to run computations on.
        beam_width (int): The number of beams to keep.
        length (int): The number of steps to generate.
        repetition_penalty (float): The penalty factor for repetition.

    Returns:
        Tuple[List[int], torch.Tensor]: The best sequence of corpus indices and the final embeddings.
    """
    model.eval()

    # Check if corpus_embeddings is empty
    if corpus_embeddings.size(0) == 0:
        raise ValueError("Corpus embeddings are empty. Cannot perform beam search on an empty corpus.")

    # Ensure start_embeddings has shape [batch_size, 1, embedding_dim]
    if start_embeddings.dim() == 1:
        start_embeddings = start_embeddings.unsqueeze(0).unsqueeze(1)  # [1, 1, embedding_dim]
    elif start_embeddings.dim() == 2:
        start_embeddings = start_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
    elif start_embeddings.dim() != 3:
        raise ValueError(f"start_embeddings must be 1D, 2D, or 3D tensor, but got {start_embeddings.dim()}D tensor.")

    start_embeddings = start_embeddings.to(device)  # [batch_size, 1, embedding_dim]
    corpus_embeddings = corpus_embeddings.to(device)  # [corpus_size, embedding_dim]
    beams = [(start_embeddings, [], 0.0)]  # (embeddings, indices, score)

    with torch.no_grad():
        for step in range(length):
            candidates = []
            for beam_embeddings, beam_indices, beam_score in beams:
                # Truncate sequence length
                if beam_embeddings.size(1) > 32:
                    beam_embeddings = beam_embeddings[:, -32:, :]  # [batch_size, 32, embedding_dim]

                next_embedding = model(beam_embeddings)  # [batch_size, 1, embedding_dim]
                next_embedding_normalized = normalize(next_embedding.squeeze(1), dim=1)  # [batch_size, embedding_dim]
                corpus_embeddings_normalized = normalize(corpus_embeddings, dim=1)  # [corpus_size, embedding_dim]

                similarities = torch.matmul(next_embedding_normalized, corpus_embeddings_normalized.t()).squeeze(0)  # [corpus_size]

                # Apply repetition penalty
                for idx in beam_indices:
                    if idx < similarities.size(0):
                        similarities[idx] = similarities[idx] / repetition_penalty

                similarities = similarities.clamp(min=1e-6)
                probabilities = torch.nn.functional.softmax(similarities, dim=0)  # [corpus_size]

                # Get top-k candidates
                top_k = min(beam_width, similarities.size(0))
                top_k_values, top_k_indices = probabilities.topk(top_k)

                for i in range(top_k):
                    idx_val = top_k_indices[i].item()
                    prob = top_k_values[i].item()
                    new_score = beam_score + np.log(prob)
                    new_embeddings = torch.cat([beam_embeddings, next_embedding], dim=1)  # [batch_size, seq_len+1, embedding_dim]
                    new_indices = beam_indices + [idx_val]
                    candidates.append((new_embeddings, new_indices, new_score))

            if not candidates:
                # No candidates to proceed with, break early
                break

            # Select top beams
            candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
            beams = candidates[:beam_width]

    if not beams:
        raise ValueError("Beam search failed to generate any beams.")

    # Select the best beam
    best_beam = beams[0]
    best_indices = best_beam[1]
    best_embeddings = best_beam[0][:, -1, :]  # [batch_size, embedding_dim]

    return best_indices, best_embeddings

def load_corpus(file_path):
    """
    Loads the corpus from a given file path.

    Args:
        file_path (str): Path to the corpus file.

    Returns:
        List[str]: List of sentences in the corpus.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus

def main():
    # Memory and CPU optimization
    gc.collect()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)

    # Model parameters
    embedding_dim = 384
    hidden_size = 256
    num_layers = 2
    model_type = "transformer"

    # Define paths
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent
    model_dir = root_dir / "backend" / "models" / "models"
    data_dir = root_dir / "data"
    model_path = model_dir / "model.pt"
    corpus_file = data_dir / "corpus_data.json"

    # Ensure directories exist
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    corpus_manager = CorpusManager(corpus_file)
    initial_prompts = ["This is the first sentence.", "Here is another sentence."]
    corpus = initial_prompts + corpus_manager.get_sentences()

    # Initialize Encoder
    encoder = Encoder()

    # Encode prompt and corpus
    print("Encoding prompts...")
    prompt_embeddings = encoder.encode_sentences(initial_prompts)  # [num_prompts, embedding_dim]

    print("Encoding corpus...")
    batch_size = 32
    corpus_chunks = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    corpus_embeddings = []

    for chunk in corpus_chunks:
        chunk_embeddings = encoder.encode_sentences(chunk)  # [batch_size, embedding_dim]
        corpus_embeddings.append(chunk_embeddings)
        gc.collect()

    corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)  # [corpus_size, embedding_dim]

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_embeddings_tensor = torch.from_numpy(prompt_embeddings).float().to(device)  # [num_prompts, embedding_dim]
    corpus_embeddings_tensor = torch.from_numpy(corpus_embeddings).float().to(device)  # [corpus_size, embedding_dim]

    # Initialize Decoder
    decoder = Decoder(embedding_dim, hidden_size, num_layers).to(device)

    # Initialize and load model
    model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type).to(device)
    if model_path.exists():
        # Set weights_only=True for enhanced security
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Starting with an untrained model.")

    # Move model to device
    model.to(device)

    # Example generation using Beam Search
    print("Starting beam search generation...")
    # Select the first prompt embedding for generation and adjust dimensions
    start_embedding = prompt_embeddings_tensor[0].unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, embedding_dim]

    # Perform beam search
    best_indices, best_embeddings = beam_search(
        model,
        start_embedding,
        corpus_embeddings_tensor,
        device,
        beam_width=5,
        length=15,  # Changed from 10 to 15
        repetition_penalty=1.2
    )

    print("Beam search completed.")

    # Decode the sequence
    print("Decoding generated embeddings...")
    decoded_results = decoder.decode_embedding(best_embeddings, corpus, corpus_embeddings_tensor, k=15)

    # Process and display the decoded sentences
    print("Generated text (Beam Search):")
    for idx, (sentence, index, similarity) in enumerate(decoded_results):
        print(f"Step {idx + 1}:")
        print(f"Sentence: {sentence}")
        print(f"Corpus Index: {index}")
        print(f"Similarity Score: {similarity:.4f}\n")

if __name__ == "__main__":
    main()
