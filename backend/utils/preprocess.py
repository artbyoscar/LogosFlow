import sys
import os

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from datasets import load_dataset
from backend.models.encoder import Encoder  # Update import path
import re

def load_data(dataset_name, split="train[:10%]"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def segment_sentences(text, max_length=200):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    result = []
    for sentence in sentences:
        while len(sentence) > max_length:
            split_index = max_length
            while split_index > 0 and sentence[split_index] not in ".?!":
                split_index -= 1
            if split_index == 0:
                split_index = max_length  # No punctuation found, split at max_length
            result.append(sentence[:split_index].strip())
            sentence = sentence[split_index:].strip()
        result.append(sentence)
    return result

def encode_and_save_embeddings(dataset, encoder, output_file):
    all_embeddings = []
    for example in dataset:
        text = example["story"]  # Adjust field name based on the dataset
        sentences = segment_sentences(text)
        embeddings = encoder.encode_sentences(sentences)
        all_embeddings.extend(embeddings)
    all_embeddings = np.array(all_embeddings)
    np.save(output_file, all_embeddings)

def main():
    dataset_name = "roc_stories"  # Or "cnn_dailymail", "3.0.0"
    output_file = "data/embeddings.npy"
    encoder = Encoder()

    dataset = load_data(dataset_name)
    encode_and_save_embeddings(dataset, encoder, output_file)

if __name__ == "__main__":
    main()