import cProfile
import torch
import numpy as np
import os
import sys
import gc
import psutil  # For monitoring CPU and memory
import time    # For timing
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.model import SimplePolicyNetwork

# CPU Optimization
torch.set_num_threads(4)  # Optimize for your CPU

def prepare_data_loader(embedding_file, batch_size, seq_length=32, val_split=0.1):
    """Loads embeddings, prepares training and validation DataLoaders."""
    try:
        embeddings = np.load(embedding_file)
    except FileNotFoundError:
        print(f"Error: Could not find embeddings file at {embedding_file}")
        print("Please ensure that 'preprocess.py' has been run successfully to generate embeddings.")
        sys.exit(1)
    embeddings = torch.from_numpy(embeddings).float()

    # Create sequences of embeddings
    sequences = []
    targets = []
    for i in range(0, len(embeddings) - seq_length):
        sequences.append(embeddings[i:i + seq_length])
        targets.append(embeddings[i + seq_length])
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    dataset = TensorDataset(sequences, targets)

    # Split into training and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Use num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def monitor_system(epoch, batch_idx, interval=10):
    """Monitors CPU usage, memory consumption, and timing."""
    if batch_idx % interval == 0:
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=None)  # Get CPU usage
        memory_info = process.memory_info()
        print(f"  Epoch: {epoch+1}, Batch: {batch_idx+1}, CPU Usage: {cpu_percent:.1f}%, Memory Usage: {memory_info.rss / 1024**2:.2f} MB")

def main():
    # Hyperparameters
    embedding_dim = 384
    hidden_size = 256
    num_layers = 2
    batch_size = 64
    initial_learning_rate = 1e-3
    num_epochs = 2 # Reduced for quicker profiling
    model_type = "transformer"
    seq_length = 32
    accumulation_steps = 4
    checkpoint_interval = 5
    val_split = 0.1
    patience = 3
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Define directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(root_dir, "backend/models/models")
    data_dir = os.path.join(root_dir, "data")

    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Data Loaders
    embeddings_file_path = os.path.join(data_dir, "embeddings.npy")
    train_loader, val_loader = prepare_data_loader(embeddings_file_path, batch_size, seq_length, val_split)

    # Model, Optimizer, Loss
    model = SimplePolicyNetwork(embedding_dim, hidden_size, num_layers, model_type, seq_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    loss_function = torch.nn.MSELoss()

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()

        for batch_idx, (input_embeddings, target_embeddings) in enumerate(train_loader):
            monitor_system(epoch, batch_idx)

            # No more mixed precision on CPU
            predicted_embeddings = model(input_embeddings)
            predicted_embeddings = predicted_embeddings / predicted_embeddings.norm(p=2, dim=-1, keepdim=True)
            target_embeddings = target_embeddings / target_embeddings.norm(p=2, dim=-1, keepdim=True)
            loss = loss_function(predicted_embeddings, target_embeddings)
            loss = loss / accumulation_steps

            # Gradient Accumulation
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f"  Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_embeddings, target_embeddings) in enumerate(val_loader):
                # No monitoring during validation for simplicity
                predicted_embeddings = model(input_embeddings)
                predicted_embeddings = predicted_embeddings / predicted_embeddings.norm(p=2, dim=-1, keepdim=True)
                target_embeddings = target_embeddings / target_embeddings.norm(p=2, dim=-1, keepdim=True)
                val_loss += loss_function(predicted_embeddings, target_embeddings).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds. Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Save final model
    final_model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    cProfile.run('main()', 'train_profile')