import numpy as np
import os

def main():
    # Construct the path to embeddings.npy relative to the script's location
    embeddings_path = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")

    # Check if the file exists
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at {embeddings_path}")
        return

    try:
        embeddings = np.load(embeddings_path)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"First embedding (first 10 elements): {embeddings[0][:10]}")
    except Exception as e:
        print(f"Error loading or inspecting embeddings: {e}")

if __name__ == "__main__":
    main()