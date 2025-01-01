import sys
import os
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from datasets import load_dataset
from backend.models.encoder import Encoder  # Import from correct location
import re

def load_data(dataset_name, split="train[:50%]"):  # Increased to 50%
    """Loads the dataset using the datasets library."""
    print(f"Loading dataset: {dataset_name} with split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    print("Dataset loaded successfully.")
    return dataset

def segment_sentences(text, max_length=200):
    """Segments text into sentences with a maximum length."""
    print("Segmenting sentences...")
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
    """Augments text using synonym replacement and back translation."""

    # Synonym augmentation
    aug_syn = naw.SynonymAug(aug_src='wordnet')
    augmented_text_syn = aug_syn.augment(text)

    # Back translation augmentation (you'll need to install models)
    # See: https://nlpaug.readthedocs.io/en/latest/augmenter/sentence/back_translation.html
    try:
        aug_bt = nas.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en'
        )
        augmented_text_bt = aug_bt.augment(text)
    except Exception as e:
        print(f"Back-translation failed: {e}")
        print("Skipping back-translation for this text.")
        augmented_text_bt = text

    return [text, augmented_text_syn, augmented_text_bt]

def encode_and_save_embeddings(dataset, encoder, output_file):
    """Encodes sentences into embeddings and saves them to a file."""
    all_embeddings = []
    print("Encoding sentences and performing data augmentation...")
    for example in dataset:
        text = example["text"]  # Access the 'text' field

        # Augment the text
        augmented_texts = augment_text(text)

        for aug_text in augmented_texts:
            sentences = segment_sentences(aug_text)
            embeddings = encoder.encode_sentences(sentences)
            all_embeddings.extend(embeddings)

    all_embeddings = np.array(all_embeddings)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Debugging Prints:
    print(f"Saving embeddings to: {output_file}")
    print(f"Current working directory: {os.getcwd()}")

    np.save(output_file, all_embeddings)
    print(f"Embeddings saved to {output_file}")

def main():
    dataset_name = "rotten_tomatoes"  # Changed to rotten_tomatoes

    # Define the absolute path to the output file
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings.npy"))
    encoder = Encoder()

    dataset = load_data(dataset_name)
    encode_and_save_embeddings(dataset, encoder, output_file)

if __name__ == "__main__":
    main()