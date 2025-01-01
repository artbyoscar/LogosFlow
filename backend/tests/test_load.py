import sys
import os

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# In test_load.py
from datasets import load_dataset

if __name__ == "__main__":
    try:
        dataset = load_dataset("roc_stories", split="train[:1%]")  # Try roc_stories again
        print("Dataset loaded successfully!")
        print(dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")