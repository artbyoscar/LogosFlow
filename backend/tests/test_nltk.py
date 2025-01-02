import nltk
import os

# Set NLTK_DATA environment variable temporarily
os.environ['NLTK_DATA'] = 'C:\\Users\\OscarNuñez/nltk_data'
print(f"NLTK_DATA is set to: {os.environ['NLTK_DATA']}")

# Add the NLTK data path to the NLTK data path list
nltk.data.path.append('C:\\Users\\OscarNuñez/nltk_data')

# Print NLTK data paths
print("NLTK data paths:")
for path in nltk.data.path:
    print(path)

# Attempt to use the tagger and tokenizer
try:
    # Download the tagger if it's not found
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
        print("averaged_perceptron_tagger found.")
    except LookupError:
        print("averaged_perceptron_tagger not found. Attempting to download...")
        nltk.download('averaged_perceptron_tagger', download_dir='C:\\Users\\OscarNuñez/nltk_data')
        print("Successfully downloaded averaged_perceptron_tagger")

    tokens = nltk.word_tokenize("This is a test sentence.")
    tags = nltk.pos_tag(tokens)
    print(tags)
except LookupError as e:
    print(f"LookupError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")