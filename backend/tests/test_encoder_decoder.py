import sys
import os

# Add the parent directory of 'backend' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from backend.models.encoder import Encoder
from backend.models.decoder import Decoder

class TestEncoderDecoder(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.sentences = [
            "This is a test sentence.",
            "Another sentence for testing.",
            "Encoding and decoding should work.",
        ]

    def test_encoding(self):
        embeddings = self.encoder.encode_sentences(self.sentences)
        self.assertEqual(len(embeddings), len(self.sentences))
        self.assertEqual(embeddings[0].shape, (384,))  # Check embedding dimension

    def test_decoding(self):
        embeddings = self.encoder.encode_sentences(self.sentences)
        decoded_sentence, _ = self.decoder.decode_embedding(embeddings[0], self.sentences)
        self.assertIn(decoded_sentence, self.sentences)  # Check if decoded sentence is in the original set

if __name__ == "__main__":
    unittest.main()