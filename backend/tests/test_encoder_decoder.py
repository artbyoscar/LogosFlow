# backend/tests/test_encoder_decoder.py

import sys
import os
import unittest
import torch
import torch.nn as nn

# Adjust sys.path to include the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.models.decoder import Decoder  # Ensure correct import path
from backend.models.encoder import Encoder  # Assuming an Encoder class exists


class TestEncoderDecoder(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        torch.manual_seed(42)

        # Define required parameters for Decoder and Encoder
        self.embedding_dim = 384
        self.hidden_size = 256
        self.num_layers = 2

        # Initialize Decoder with required arguments
        self.decoder = Decoder(
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

        # Initialize Encoder with required arguments (if applicable)
        self.encoder = Encoder()

        # Create sample input data
        self.sample_embeddings = ["This is a test sentence."] * 10  # Example input as list of strings
        self.device = torch.device("cpu")

    def test_decoder_forward_pass(self):
        """
        Test that the Decoder's forward method processes input correctly.
        """
        output = self.decoder(torch.randn(10, self.embedding_dim))
        # Adjust the expected shape based on Decoder implementation
        expected_shape = (10, self.embedding_dim)  # Adjusted expected shape
        self.assertEqual(output.shape, expected_shape, "Decoder output shape mismatch.")

    def test_encoder_forward_pass(self):
        """
        Test that the Encoder's forward method processes input correctly.
        """
        output = self.encoder.encode_sentences(self.sample_embeddings)
        # Adjust the expected shape based on Encoder implementation
        expected_shape = (10, self.embedding_dim)  # Adjusted expected shape
        self.assertEqual(output.shape, expected_shape, "Encoder output shape mismatch.")

    def test_encoder_decoder_integration(self):
        """
        Test the integration between Encoder and Decoder.
        """
        encoded = self.encoder.encode_sentences(self.sample_embeddings)
        decoded = self.decoder(torch.tensor(encoded))
        # Adjust the expected shape based on integration
        expected_shape = (10, self.embedding_dim)  # Adjusted expected shape
        self.assertEqual(decoded.shape, expected_shape, "Integrated Encoder-Decoder output shape mismatch.")

    def test_decoder_with_invalid_input(self):
        """
        Test that the Decoder raises an error when provided with invalid input dimensions.
        """
        invalid_input = torch.randn(10, self.embedding_dim + 10)  # Mismatched embedding_dim
        with self.assertRaises(
            ValueError, msg="Decoder should raise ValueError with invalid input dimensions."
        ):
            self.decoder(invalid_input)


if __name__ == '__main__':
    unittest.main()
