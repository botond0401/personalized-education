import torch
import torch.nn as nn
import hashlib
import numpy as np


# Transform to deterministic random vector function


def transform_to_random_vector(problem_id, correct, dim=10):
    unique_seed = f"{problem_id}_{correct}"
    seed = int(hashlib.md5(unique_seed.encode()).hexdigest(), 16) % (10**8)
    np.random.seed(seed)
    random_vector = np.random.randn(dim)
    return random_vector

# Custom embedding layer to handle (problem_id, correct) pairs
class CustomEmbeddingLayer(nn.Module):
    def __init__(self, dim=10):
        super(CustomEmbeddingLayer, self).__init__()
        self.dim = dim

    def forward(self, input_pairs):
        """
        Input:
            input_pairs (Tensor): shape (batch_size, seq_len, 2), each entry is (problem_id, correct)
        Output:
            Tensor of shape (batch_size, seq_len, dim) with embeddings
        """
        batch_size, seq_len, _ = input_pairs.size()
        embedded = torch.zeros(batch_size, seq_len, self.dim, dtype=torch.float32)
        
        for i in range(batch_size):
            for j in range(seq_len):
                problem_id, correct = input_pairs[i, j]
                problem_id = int(problem_id.item())
                correct = int(correct.item())
                random_vector = transform_to_random_vector(problem_id, correct, self.dim)
                embedded[i, j] = torch.tensor(random_vector, dtype=torch.float32)
        
        return embedded