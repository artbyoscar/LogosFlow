import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class SimplePolicyNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, model_type="lstm"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type

        self.input_projection = nn.Linear(embedding_dim, hidden_size)

        if model_type == "lstm":
            self.core_model = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "gru":
            self.core_model = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "transformer":
            config = AutoConfig.from_pretrained("distilgpt2")
            config.n_embd = hidden_size
            config.n_layer = num_layers
            config.n_head = 8
            config.n_positions = 512
            config.n_ctx = 512
            self.core_model = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError("Invalid model type")

        self.output_projection = nn.Linear(hidden_size, embedding_dim)

    def forward(self, embeddings):
        x = self.input_projection(embeddings)

        if self.model_type in ["lstm", "gru"]:
            outputs, _ = self.core_model(x)
            # Get the output of the last step of the sequence
            last_output = outputs[:, -1, :]
        elif self.model_type == "transformer":
             # The causal LM model expects input_ids, not embeddings.
             # We'll just use a dummy tensor of zeros here as a placeholder.
            dummy_input_ids = torch.zeros_like(x, dtype=torch.long).to(x.device)
            outputs = self.core_model(inputs_embeds=x, labels=dummy_input_ids).logits
            # Use mean pooling for sentence-level prediction
            last_output = outputs.mean(dim=1)  # Average across the sequence length
        else:
            raise ValueError("Invalid model type")

        predicted_embedding = self.output_projection(last_output)
        return predicted_embedding