import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .dkt_embedding import CustomEmbedding


class DKT(nn.Module):
    def __init__(self, num_skills, num_other, embed_dim, hid_size, num_hid_layers, drop_prob):
        super(DKT, self).__init__()

        self.num_skills = num_skills
        self.num_other = num_other
        self.embed_dim = embed_dim
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.drop_prob = drop_prob

        # Custom embedding layer
        self.embedding = CustomEmbedding(num_skills, embed_dim)

        # RNN layer
        self.rnn = nn.LSTM(embed_dim + num_other, hid_size, num_hid_layers, batch_first=True)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=drop_prob)

        # Output layer mapping hidden states to probabilities
        self.out = nn.Linear(hid_size, num_skills)

        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()

    def forward(self, skills, other, lengths):
        """
        Forward pass for the DKT model.

        Args:
            inputs (Tensor): Input sequence of shape (batch_size, seq_len, 2), where each entry is (problem_id, correct).
            lengths (Tensor): Lengths of sequences (batch_size).

        Returns:
            Tensor: Output probabilities of shape (batch_size, seq_len, num_items).
        """
        # Embed the input sequence
        embedded = self.embedding(skills)

        concatenated = torch.cat((embedded, other), dim=-1)  # (batch_size, seq_length, embed_dim+other_dim)

        # Pack the padded sequence for the RNN
        packed_embedded = pack_padded_sequence(concatenated, lengths, batch_first=True, enforce_sorted=False)

        # RNN processing
        packed_output, _ = self.rnn(packed_embedded)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout
        output = self.dropout(output)

        # Output layer for probabilities
        logits = self.out(output)

        # Sigmoid to convert logits to probabilities
        probabilities = self.sigmoid(logits)

        return probabilities
    
    def get_embedding(self, skill_id):
        one_hot = torch.zeros(1, 1, self.num_skills)
        one_hot[0, 0, skill_id] = 1
        embedding = self.embedding(one_hot)
        return embedding.squeeze(0).squeeze(0).detach().numpy()
