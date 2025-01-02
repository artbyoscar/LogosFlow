import nltk
import os

# Set NLTK_DATA environment variable temporarily
os.environ['NLTK_DATA'] = 'C:\\Users\\OscarNuñez/nltk_data'

# Add the NLTK data path to the NLTK data path list
nltk.data.path.append('C:\\Users\\OscarNuñez/nltk_data')

try:
    # Try tagging with explicit language specification
    nltk.data.find('taggers/averaged_perceptron_tagger')
    tokens = nltk.word_tokenize("This is a test sentence.")
    tags = nltk.pos_tag(tokens, lang='eng')
    print("Tagging successful:", tags)

except LookupError as e:
    print(f"LookupError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")