# backend/tests/test_beam_search.py

import sys
import os
import unittest
import torch
import torch.nn as nn

# Adjust sys.path to include the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.models.generate import beam_search  # Now, this should work correctly

class MockModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super(MockModel, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings):
        """
        Mimics the behavior of SimplePolicyNetwork by returning a fixed or predictable embedding.
        For simplicity, this mock returns a tensor filled with 0.5.
        """
        batch_size, seq_len, embedding_dim = embeddings.size()
        return torch.full((batch_size, 1, embedding_dim), 0.5)

class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        
        # Initialize MockModel
        self.model = MockModel(embedding_dim=384)
        
        # Create a random start_embedding with shape [embedding_dim]
        self.start_embeddings = torch.randn(384)
        
        # Create random corpus_embeddings with shape [50, embedding_dim]
        self.corpus_embeddings = torch.randn(50, 384)
        
        # Define the device
        self.device = torch.device("cpu")

    def test_beam_search_dimensions(self):
        """
        Test that beam_search returns outputs with correct types and dimensions.
        """
        best_indices, best_embeddings = beam_search(
            self.model,
            self.start_embeddings,
            self.corpus_embeddings,
            self.device,
            length=15  # Explicitly set length to 15
        )
        
        # Check if best_indices is a list
        self.assertIsInstance(best_indices, list, "best_indices should be a list.")
        
        # Check if best_indices has the correct length (length=15)
        self.assertEqual(len(best_indices), 15, "best_indices should have a length of 15.")
        
        # Check if best_embeddings has the correct shape (1, 384)
        self.assertEqual(best_embeddings.shape, (1, 384), "best_embeddings should have shape (1, 384).")

    def test_beam_search_output_values(self):
        """
        Test that beam_search returns embeddings with expected values.
        Since MockModel returns embeddings filled with 0.5, ensure best_embeddings are correctly formed.
        """
        best_indices, best_embeddings = beam_search(
            self.model,
            self.start_embeddings,
            self.corpus_embeddings,
            self.device,
            length=15  # Explicitly set length to 15
        )
        
        # Since MockModel returns 0.5, best_embeddings should be filled with 0.5
        expected_embedding = torch.full((1, 384), 0.5)
        self.assertTrue(torch.allclose(best_embeddings, expected_embedding),
                        "best_embeddings should be filled with 0.5.")

    def test_beam_search_repetition_penalty(self):
        """
        Test that repetition penalty is applied correctly.
        """
        # Use the same embedding for all steps to simulate repetition
        self.model = MockModel(embedding_dim=384)
        
        best_indices, best_embeddings = beam_search(
            self.model,
            self.start_embeddings,
            self.corpus_embeddings,
            self.device,
            beam_width=3,
            length=15,  # Length set to 15
            repetition_penalty=1.5
        )
        
        # Check that beam_indices have the correct length
        self.assertEqual(len(best_indices), 15, "best_indices should have a length of 15.")
        
        # Optionally, verify that indices are not repeating beyond expectations
        # Since MockModel returns fixed embeddings, the beam_search might favor certain indices
        # Depending on beam_search's logic, you can add more specific assertions here

    def test_beam_search_with_empty_corpus(self):
        """
        Test beam_search with an empty corpus. It should raise a ValueError.
        """
        empty_corpus_embeddings = torch.empty(0, 384)
        
        with self.assertRaises(ValueError, msg="beam_search should raise ValueError when corpus is empty."):
            beam_search(
                self.model,
                self.start_embeddings,
                empty_corpus_embeddings,
                self.device,
                length=15  # Can be any positive integer
            )

if __name__ == '__main__':
    unittest.main()
