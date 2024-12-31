import torch
import numpy as np
from model import SimplePolicyNetwork
from decoder import Decoder

def generate_sequence(model, start_embedding, length, corpus, corpus_embeddings, device):
    model.eval()
    generated_embeddings = [start_embedding]
    current_embedding = start_embedding.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            next_embedding = model(current_embedding)
            generated_embeddings.append(next_embedding.squeeze(0))
            current_embedding = next_embedding

    decoder = Decoder()
    generated_sentences = []
    for emb in generated_embeddings:
        decoded_sentence, _ = decoder.decode_embedding(emb, corpus, corpus_embeddings)
        generated_sentences.append(decoded_sentence)

    return generated_sentences

def main():
    embedding_dim = 384
    hidden_size = 128
    num_layers = 2
    model_type = "transformer"  # Changed model type to transformer
    model_path = "models/model.pt"  # Replace with your saved model path
    prompt = ["This is the first sentence.", "Here is another sentence."]
    corpus = prompt # adjust as necessary

    # Load the trained model
    model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Ensure model is loaded to CPU
    device = torch.device("cpu")
    model.to(device)

    # Encode the prompt
    encoder = Encoder()
    prompt_embeddings = encoder.encode_sentences(prompt)
    corpus_embeddings = encoder.encode_sentences(corpus)
    start_embedding = torch.tensor(prompt_embeddings, dtype=torch.float32)

    # Generate a sequence of embeddings
    generated_sentences = generate_sequence(model, start_embedding[-1], 5, corpus, corpus_embeddings, device)  # Generate 5 more sentences

    # Print the generated text
    print("Generated Text:")
    for sentence in generated_sentences:
        print(sentence)

if __name__ == "__main__":
    main()