import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dkt_embedding import CustomEmbeddingLayer

# Define the DKT model


class DKT(nn.Module):
    def __init__(self, num_items, embed_dim, hid_size, num_hid_layers, drop_prob):
        super(DKT, self).__init__()
        
        # Custom embedding layer for (problem_id, correct) pairs
        self.embedding = CustomEmbeddingLayer(embed_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embed_dim, hid_size, num_hid_layers, batch_first=True)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=drop_prob)
        
        # Output layer mapping hidden states to probabilities
        self.out = nn.Linear(hid_size, num_items)
        
        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, lengths):
        # Embed the input sequence
        embedded = self.embedding(inputs)
        
        # Pack the padded sequence for the RNN
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        # RNN processing
        packed_output, _ = self.rnn(packed_embedded)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout
        output = self.dropout(output)

        # Output layer for probabilities
        output = self.out(output)

        # Sigmoid to convert logits to probabilities
        output = self.sigmoid(output)

        mask = torch.arange(output.size(1)).expand(len(lengths), output.size(1)) < lengths.unsqueeze(1)

        # Expand mask for the number of items and apply it to the output
        mask = mask.unsqueeze(-1).expand_as(output)  # Shape: (batch_size, max_seq_length, num_items)
        masked_output = output * mask  # Zero out the padded positions

        return masked_output # Adjust shape for loss function compatibility
    