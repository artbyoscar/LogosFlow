import sys
import os
import nlpaug.augmenter.word as naw
# Note: We've removed the import for 'nlpaug.augmenter.sentence' as it's not used

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from datasets import load_dataset
from backend.models.encoder import Encoder
import re

def load_data(dataset_name, split="train[:50%]"):
    """Loads the dataset using the datasets library."""
    print(f"Loading dataset: {dataset_name} with split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    print("Dataset loaded successfully.")
    return dataset

def segment_sentences(text, max_length=200):
    """Segments text into sentences with a maximum length."""
    print("Segmenting sentences...")
    # Handle the case where text is a list (from previous augmentation)
    if isinstance(text, list):
        text = ' '.join(text)  # Join the list into a single string

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
    print("Sentences segmented.")
    return result

def augment_text(text):
    """Augments text using synonym replacement."""

    # Synonym augmentation
    aug_syn = naw.SynonymAug(aug_src='wordnet')
    augmented_text_syn = aug_syn.augment(text)

    return [text, augmented_text_syn]  # Only return original and synonym-augmented text

def encode_and_save_embeddings(dataset, encoder, output_file):
    """Encodes sentences into embeddings and saves them to a file."""
    all_embeddings = []
    print("Encoding sentences and performing data augmentation...")
    for example in dataset:
        text = example["text"]

        # Augment the text
        augmented_texts = augment_text(text)

        for aug_text in augmented_texts:
            sentences = segment_sentences(aug_text)
            embeddings = encoder.encode_sentences(sentences)
            all_embeddings.extend(embeddings)

    all_embeddings = np.array(all_embeddings)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving embeddings to: {output_file}")
    print(f"Current working directory: {os.getcwd()}")

    np.save(output_file, all_embeddings)
    print(f"Embeddings saved to {output_file}")

def main():
    dataset_name = "rotten_tomatoes"

    # Define the absolute path to the output file
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings.npy"))
    encoder = Encoder()

    dataset = load_data(dataset_name)
    encode_and_save_embeddings(dataset, encoder, output_file)

if __name__ == "__main__":
    main()