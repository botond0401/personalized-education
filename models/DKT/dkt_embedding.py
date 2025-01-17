import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    def __init__(self, num_skills, embed_dim):
        super(CustomEmbedding, self).__init__()
        # Linear layer for embedding
        self.linear = nn.Linear(num_skills, embed_dim)
        # Tanh activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Perform the linear transformation
        embedded = self.linear(x)

        # Calculate the number of 1s (sum of the input binary vector)
        num_ones = torch.sum(x, dim=-1, keepdim=True)

        # Broadcast num_ones to the shape of embedded: (batch_size, seq_len, embed_dim)
        num_ones = num_ones.expand(-1, -1, embedded.size(-1))

        # Avoid division by zero
        num_ones = torch.max(num_ones, torch.ones_like(num_ones))

        # Divide the embedded vector by the number of 1s
        embedded /= num_ones

        # Apply the Tanh activation
        return self.tanh(embedded)
