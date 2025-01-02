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

def beam_search(model, start_embeddings, corpus_embeddings, beam_width=5, length=10, repetition_penalty=1.2):
    model.eval()
    beams = [(start_embeddings, [], 0.0)]  # (embeddings, indices, score)

    with torch.no_grad():
        for _ in range(length):
            candidates = []
            for beam_embeddings, beam_indices, beam_score in beams:
                next_embedding = model(beam_embeddings.unsqueeze(0))

                print("Shape of next_embedding:", next_embedding.shape)  # Debug print

                # next_embedding shape: [1, embedding_dim] (no sequence length dimension)
                # Remove the slicing:
                # next_embedding = next_embedding[:, -1, :]

                # If needed, squeeze the extra dimension:
                if next_embedding.shape[0] == 1:
                    next_embedding = next_embedding.squeeze(0)  # Now [embedding_dim]

                similarities = torch.matmul(
                    torch.nn.functional.normalize(next_embedding, dim=0),
                    torch.nn.functional.normalize(corpus_embeddings, dim=1).t()
                )

                # Apply repetition penalty to similarities
                for idx in beam_indices[-5:]:  # Penalize the last 5 tokens
                    similarities[:, idx] /= repetition_penalty

                top_k_values, top_k_indices = similarities.topk(beam_width)

                for prob, idx in zip(top_k_values.squeeze(), top_k_indices.squeeze()):
                    new_score = beam_score + torch.log(prob)
                    candidates.append((
                        torch.cat([beam_embeddings, next_embedding.unsqueeze(0)], dim=0),  # Add a dimension to next_embedding
                        beam_indices + [idx.item()],
                        new_score
                    ))

            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

    # Get the best beam (highest score)
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

    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(root_dir, "backend", "models", "models")
    data_dir = os.path.join(root_dir, "data")
    model_path = os.path.join(model_dir, "model.pt")

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Load corpus using CorpusManager
    corpus_manager = CorpusManager(os.path.join(data_dir, "corpus_data.json"))
    prompt = ["This is the first sentence.", "Here is another sentence."]
    corpus = prompt + corpus_manager.get_sentences()

    # Load and configure model
    try:
        model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Starting with an untrained model.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        return

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

    start_embeddings = prompt_embeddings_tensor
    current_embeddings = start_embeddings.unsqueeze(0).to(device)

    # Or use beam search
    best_indices, best_embeddings = beam_search(
        model,
        start_embeddings,
        corpus_embeddings_tensor,
        beam_width=5,
        length=15,
        repetition_penalty=1.2  # Adjust if needed
    )

    # Decode the best sequence from beam search
    decoder = Decoder(embedding_dim, hidden_size, num_layers)
    generated_sentences = []

    for emb in best_embeddings:
        decoded_sentence, _ = decoder.decode_embedding(emb.unsqueeze(0), corpus, corpus_embeddings_tensor, k=15)
        generated_sentences.append(decoded_sentence[0])

    generated_text = " ".join(generated_sentences)
    print("Generated text (Beam Search):", generated_text)

if __name__ == "__main__":
    main()