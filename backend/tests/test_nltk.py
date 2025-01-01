import nltk
import os

# Check if NLTK_DATA environment variable is set
if 'NLTK_DATA' in os.environ:
    print(f"NLTK_DATA is set to: {os.environ['NLTK_DATA']}")
else:
    print("NLTK_DATA is not set.")
    # Optionally, set it here if you want to test without environment variables
    # os.environ['NLTK_DATA'] = 'C:\\Users\\OscarNuñez/nltk_data'

# Add the NLTK data path to the NLTK data path list
nltk.data.path.append('C:\\Users\\OscarNuñez/nltk_data')

# Print NLTK data paths
print("NLTK data paths:")
for path in nltk.data.path:
    print(path)

# Attempt to download the tagger
try:
    nltk.download('averaged_perceptron_tagger', download_dir='C:\\Users\\OscarNuñez/nltk_data')
    print("Successfully downloaded averaged_perceptron_tagger")
except Exception as e:
    print(f"Failed to download: {e}")

# Attempt to use the tagger
try:
    tokens = nltk.word_tokenize("This is a test sentence.")
    tags = nltk.pos_tag(tokens)
    print(tags)
except LookupError as e:
    print(f"LookupError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")