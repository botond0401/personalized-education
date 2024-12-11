import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Define the DKT model
class DKT(nn.Module):
    def __init__(self, num_skills, num_other, embed_dim, hid_size, num_hid_layers, drop_prob):
        super(DKT, self).__init__()

        # Custom embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(num_skills, embed_dim),
            nn.Tanh()
            )

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

        # Mask padded positions
        mask = torch.arange(probabilities.size(1)).expand(len(lengths), probabilities.size(1)) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(probabilities)  # Shape: (batch_size, seq_len, num_items)
        mask = mask.to(probabilities.device)
        masked_output = probabilities * mask.float()  # Zero out the padded positions

        return masked_output
    