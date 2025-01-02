import torch
import torch.nn as nn
import math
import gc

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SimplePolicyNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, model_type="lstm", seq_length=32):
        super().__init__()
        torch.set_num_threads(4)

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.seq_length = seq_length

        self.input_projection = nn.Linear(embedding_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.input_projection.weight)
        self.dropout = nn.Dropout(0.2)

        if model_type == "lstm":
            self.core_model = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        elif model_type == "gru":
            self.core_model = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0
            )
        elif model_type == "transformer":
            self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_length)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dropout=0.2,
                batch_first=True,
                dim_feedforward=hidden_size * 2
            )
            self.core_model = nn.TransformerEncoder(encoder_layer, num_layers)
        else:
            raise ValueError("Invalid model type")

        self.output_projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.output_projection.weight)
        self.output_dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.tanh = nn.Tanh()

        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, embeddings):
        gc.collect()
        embeddings = embeddings.to(self.device)
        # Ensure embeddings is 3D: (batch_size, seq_len, embedding_dim)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # Add sequence dimension if missing

        x = self.input_projection(embeddings)
        x = self.dropout(x)

        if self.model_type in ["lstm", "gru"]:
            outputs, _ = self.core_model(x)
            last_output = outputs[:, -1, :]
        elif self.model_type == "transformer":
            x = self.pos_encoder(x)
            outputs = self.core_model(x)
            last_output = outputs[:, -1, :]

        x = self.output_dropout(last_output)
        x = self.output_projection(x)
        x = self.layer_norm(x)
        predicted_embedding = self.tanh(x)

        return predicted_embedding