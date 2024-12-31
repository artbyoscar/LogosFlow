import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import SimplePolicyNetwork

def prepare_data_loader(embedding_file, batch_size):
    embeddings = np.load(embedding_file)
    embeddings = torch.from_numpy(embeddings).float()
    dataset = TensorDataset(embeddings[:-1], embeddings[1:])  # Predict next embedding
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def main():
    # Hyperparameters
    embedding_dim = 384
    hidden_size = 128
    num_layers = 1
    batch_size = 16 # Reduced batch size
    learning_rate = 1e-3
    num_epochs = 10

    data_loader = prepare_data_loader("data/embeddings.npy", batch_size)
    model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type="transformer")  # Changed model type to transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, (input_embeddings, target_embeddings) in enumerate(data_loader):
            optimizer.zero_grad()
            predicted_embedding = model(input_embeddings)
            loss = loss_function(predicted_embedding, target_embeddings)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()