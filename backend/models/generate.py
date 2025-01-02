import torch
import numpy as np
import os
import sys
import gc
from torch.nn.functional import cosine_similarity, normalize
from pathlib import Path

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.model import SimplePolicyNetwork
from backend.models.decoder import Decoder
from backend.models.encoder import Encoder
from backend.utils.corpus_manager import CorpusManager

def apply_repetition_penalty(probabilities, generated_indices, penalty=1.2):
    if len(generated_indices) > 0:
        for idx in generated_indices[-5:]:
            probabilities[idx] = probabilities[idx] / penalty
        return probabilities / probabilities.sum()
    return probabilities


def generate_sequence(model, start_embeddings, length, corpus, corpus_embeddings, device, k=15, temperature=1.5, top_p=0.9):
    model.eval()
    generated_embeddings = []
    closest_indices = []
    torch.set_num_threads(4)
    gc.collect()

    current_embeddings = start_embeddings.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            if current_embeddings.size(1) > 32:
                current_embeddings = current_embeddings[:, -32:, :]
            
            next_embedding = model(current_embeddings)
            next_embedding_normalized = torch.nn.functional.normalize(next_embedding.squeeze(0), dim=0)
            corpus_embeddings_normalized = torch.nn.functional.normalize(corpus_embeddings, dim=1)

            # Compute similarities and apply temperature
            similarities = torch.matmul(next_embedding_normalized.unsqueeze(0), 
                                        corpus_embeddings_normalized.t()).squeeze(0)
            scaled_similarities = similarities / temperature
            
            # Nucleus sampling
            probabilities = torch.nn.functional.softmax(scaled_similarities, dim=0)
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            probabilities[indices_to_remove] = 0
            probabilities = probabilities / probabilities.sum()

            # Apply repetition penalty
            probabilities = apply_repetition_penalty(probabilities, closest_indices)

            # Sample from filtered distribution
            top_k_values, top_k_indices = probabilities.topk(min(k, len(corpus)))
            top_k_probs = top_k_values / top_k_values.sum()
            selected_idx = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
            
            closest_indices.append(selected_idx)
            generated_embeddings.append(next_embedding.squeeze(0).cpu())

            next_embedding = next_embedding.unsqueeze(1)
            current_embeddings = torch.cat([current_embeddings, next_embedding], dim=1)
            
            if _ % 5 == 0:
                gc.collect()

    return generated_embeddings, closest_indices

def beam_search(model, start_embeddings, corpus_embeddings, device, beam_width=5, length=10, repetition_penalty=1.2):
    model.eval()

    # Ensure start_embeddings is 3D: (batch_size, seq_len, embedding_dim)
    if start_embeddings.dim() == 2:
        start_embeddings = start_embeddings.unsqueeze(1)  # Add sequence dimension if missing

    start_embeddings = start_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)
    beams = [(start_embeddings, [], 0.0)]  # (embeddings, indices, score)

    with torch.no_grad():
        for _ in range(length):
            candidates = []
            for beam_embeddings, beam_indices, beam_score in beams:
                # Ensure beam_embeddings is 3D
                if beam_embeddings.dim() == 2:
                    beam_embeddings = beam_embeddings.unsqueeze(1)

                next_embedding = model(beam_embeddings)

                # Check if next_embedding has a sequence dimension
                if next_embedding.dim() == 3:
                    next_embedding_normalized = normalize(next_embedding[:, -1, :], dim=1)
                else:  # next_embedding is 2D
                    next_embedding_normalized = normalize(next_embedding, dim=1)  # Normalize along the embedding dimension

                corpus_embeddings_normalized = normalize(corpus_embeddings, dim=1)

                similarities = torch.matmul(next_embedding_normalized, corpus_embeddings_normalized.t())

                if similarities.dim() > 1:
                    similarities = similarities.squeeze(0)

                # Apply repetition penalty
                for idx in beam_indices[-5:]:
                    similarities[idx] /= repetition_penalty

                similarities += 1e-8  # Add a small value to avoid log(0)
                similarities = similarities.clamp(min=1e-6)  # Prevent very small values

                # Get top-k, but handle cases where there are fewer than beam_width elements
                num_candidates = min(beam_width, similarities.size(0))
                top_k_values, top_k_indices = similarities.topk(num_candidates)

                # Only consider unique indices
                unique_indices = torch.unique(top_k_indices)
                
                # Create new candidates for each of the unique top_k results
                for idx in unique_indices:
                    idx_val = idx.item()
                    
                    # Find the corresponding probability for the unique index
                    prob_index = (top_k_indices == idx).nonzero(as_tuple=True)[0]
                    prob = top_k_values[prob_index].max()  # Use max to handle potential duplicates in top_k_values

                    new_score = beam_score + torch.log(prob)

                    # Handle 2D or 3D tensor appropriately
                    if next_embedding.dim() == 3:
                        new_embedding_to_cat = next_embedding[:, -1:, :].clone().detach()
                    else:
                        new_embedding_to_cat = next_embedding.unsqueeze(1).clone().detach()

                    new_embeddings = torch.cat([beam_embeddings, new_embedding_to_cat], dim=1)
                    candidates.append((
                        new_embeddings,
                        beam_indices + [idx_val],
                        new_score
                    ))

            # Select top-k beams
            candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
            beams = candidates[:beam_width]

    best_beam = beams[0]
    best_indices = best_beam[1]
    best_embeddings = best_beam[0]

    return best_indices, best_embeddings

def load_corpus(file_path):
    with open(file_path, 'r') as f:
        corpus = [line.strip() for line in f]
    return corpus

def main():
    # Memory and CPU optimization
    gc.collect()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)  # Disable gradients for inference

    # Model parameters optimized for your hardware
    embedding_dim = 384
    hidden_size = 256
    num_layers = 2
    model_type = "transformer"

    # Define paths (updated to be relative to the script's location)
    script_dir = Path(__file__).parent  # Gets the directory of the current script
    root_dir = script_dir.parent.parent  # Moves up two levels to the 'LogosFlow' directory
    model_dir = root_dir / "backend" / "models" / "models"  # Path to the model directory
    data_dir = root_dir / "data" # Path to the data directory
    model_path = model_dir / "model.pt" # Path to the model file

    # Ensure directories exist
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus using CorpusManager
    corpus_manager = CorpusManager(data_dir / "corpus_data.json")
    prompt = ["This is the first sentence.", "Here is another sentence."]
    corpus = prompt + corpus_manager.get_sentences()

    # Load and configure model
    model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}. Starting with an untrained model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode prompt and corpus in batches
    encoder = Encoder()
    prompt_embeddings = encoder.encode_sentences(prompt)

    batch_size = 32
    corpus_chunks = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    corpus_embeddings = []

    for chunk in corpus_chunks:
        chunk_embeddings = encoder.encode_sentences(chunk)
        corpus_embeddings.append(chunk_embeddings)
        gc.collect()  # Clear memory after each chunk

    corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)

    # Convert to tensors
    prompt_embeddings_tensor = torch.from_numpy(prompt_embeddings).float().to(device)
    corpus_embeddings_tensor = torch.from_numpy(corpus_embeddings).float().to(device)

    # Generate text using beam search
    best_indices, best_embeddings = beam_search(
        model,
        prompt_embeddings_tensor,
        corpus_embeddings_tensor,
        device,
        beam_width=5,
        length=15,
        repetition_penalty=1.2  # Adjust if needed
    )

    # Decode the best sequence from beam search
    decoder = Decoder(embedding_dim, hidden_size, num_layers)
    generated_sentences = []

    for emb in best_embeddings:
        decoded_sentence, _, _ = decoder.decode_embedding(emb.unsqueeze(0), corpus, corpus_embeddings_tensor, k=15)
        generated_sentences.append(decoded_sentence)

    generated_text = " ".join(generated_sentences)
    print("Generated text (Beam Search):", generated_text)

if __name__ == "__main__":
    main()