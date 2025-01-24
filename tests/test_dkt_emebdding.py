import torch
import pytest
import sys
import os

# Add correct paths
models_path = os.path.join(os.getcwd(), 'models')

# Append to sys.path\n",
sys.path.append(models_path)

from DKT.dkt_embedding import CustomEmbedding

# Tests for the CustomEmbedding class
def test_custom_embedding_shape():
    num_skills = 5
    embed_dim = 3
    batch_size = 2
    seq_len = 3

    model = CustomEmbedding(num_skills, embed_dim)

    zero_input = torch.zeros(batch_size, seq_len, num_skills)
    output = model(zero_input)
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len,embed_dim)}, but got {output.shape}"

def test_custom_embedding_nans():
    num_skills = 5
    embed_dim = 3
    batch_size = 2
    seq_len = 3

    model = CustomEmbedding(num_skills, embed_dim)
    one_input = torch.ones(batch_size, seq_len, num_skills)
    output_ones = model(one_input)
    assert not torch.isnan(output_ones).any(), "Output contains NaNs!"

def test_custom_embedding_output_match():
    # Define the input tensor values as per your request
    num_skills = 3
    input_tensor = torch.tensor([[[1, 0, 0], [0, 0, 1], [1, 0, 1]]], dtype=torch.float32)
    embed_dim = 2

    # Create the model with embed_dim = 2
    model = CustomEmbedding(num_skills=num_skills, embed_dim=embed_dim)

    # Get the output from the model
    output = model(input_tensor)

    # Extract embeddings for the first two inputs and calculate their average
    embedding_1 = output[0][0]
    embedding_2 = output[0][1]
    expected_embedding_3 = (embedding_1 + embedding_2) / 2

    # Check if the embedding of the third input is close to the average of the first two
    assert torch.allclose(output[0][2], expected_embedding_3, atol=1e-6), "The third input's embedding is not the average of the first two embeddings!"

# Run all the tests
if __name__ == "__main__":
    pytest.main()
