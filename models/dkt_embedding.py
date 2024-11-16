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

class CustomEmbeddingLayer(nn.Module):
    def __init__(self, num_items, dim=10):
        super(CustomEmbeddingLayer, self).__init__()
        self.num_items = num_items
        self.dim = dim

        # Create a trainable embedding matrix with deterministic initialization
        self.embedding = nn.Embedding(num_items * 2 + 1, dim, padding_idx=0)  # *2 for (problem_id, correct)

        # Initialize embeddings deterministically
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        # Start with a fixed random vector for the padding index
        random_vector = transform_to_random_vector(0, 0, self.dim)
        embeddings = [torch.tensor(random_vector, dtype=torch.float32)]

        # Initialize embeddings for each (problem_id, correct) pair
        for problem_id in range(1, self.num_items + 1):
            for correct in [1, 0]:  # Correct can be 0 or 1
                random_vector = transform_to_random_vector(problem_id, correct, self.dim)
                embeddings.append(torch.tensor(random_vector, dtype=torch.float32))

        # Stack all embeddings and copy them to the embedding layer's weights
        embeddings = torch.stack(embeddings)  # Shape: (num_items * 2, dim)
        self.embedding.weight.data.copy_(embeddings)

    def forward(self, input_pairs):
        """
        Input:
            input_pairs (Tensor): shape (batch_size, seq_len, 2), each entry is (problem_id, correct)
        Output:
            Tensor of shape (batch_size, seq_len, dim) with embeddings
        """
        # Extract problem_id and correct values
        problem_ids = input_pairs[..., 0]  # Shape: (batch_size, seq_len)
        correct = input_pairs[..., 1]      # Shape: (batch_size, seq_len)

        # Ensure valid problem_ids and correctness values
        assert torch.all(problem_ids >= 0) and torch.all(problem_ids <= self.num_items), \
            f"problem_ids out of range! Min: {problem_ids.min()}, Max: {problem_ids.max()}, num_items: {self.num_items}"
        
        assert torch.all((correct == 0) | (correct == 1)), "Correct values should be either 0 or 1"

        # Map (problem_id, correct) pairs to embedding indices
        indices = problem_ids * 2 - correct.long()

        # Ensure indices are within the bounds of the embedding matrix
        assert torch.all(indices >= 0) and torch.all(indices <= self.num_items * 2), \
            f"indices out of range! Min: {indices.min()}, Max: {indices.max()}, num_items*2: {self.num_items * 2}"

        # Return the embeddings based on the indices
        return self.embedding(indices)