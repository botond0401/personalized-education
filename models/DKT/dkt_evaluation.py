import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_DKT_loss(predictions_all, answers_with_labels, lengths):
    """
    Calculate the binary cross-entropy loss for a sequence of predictions and labels,
    while ignoring padded positions based on the given lengths.

    Args:
        predictions_all (Tensor): The model's predicted values with shape (batch_size, seq_len, num_skills).
        answers_with_labels (Tensor): Ground truth tensor with shape (batch_size, seq_len, num_skills + 1),
                                      where the last dimension contains problem IDs and correctness labels.
                                      The last element of each entry indicates whether the response was correct (1) or not (0).
        lengths (Tensor): A tensor of shape (batch_size,) indicating the lengths of each sequence in the batch.

    Returns:
        loss (Tensor): The total binary cross-entropy loss for the batch, ignoring padded positions.
    """
    # Extract the result (predictions) and labels from the inputs
    result, labels = _transform_to_correct_predictions(predictions_all, answers_with_labels)

    # Create a mask based on the sequence lengths, where 1 represents a valid position and 0 represents padding
    batch_size, seq_len = result.size()  # Assuming result is (batch_size, seq_len)
    mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)
    mask = mask.float()  # Convert to float for later multiplication

    # Apply the mask to the result and labels to ignore padded positions
    masked_result = result * mask
    masked_labels = labels * mask

    # Compute the binary cross-entropy loss for each sequence element
    loss = F.binary_cross_entropy(masked_result, masked_labels, reduction='none')

    # Average the loss over the non-padded positions
    masked_loss = loss.sum() / mask.sum()  # Normalize by the number of valid (non-padded) positions

    return masked_loss


def calculate_auc(predictions_all, answers_with_labels, lengths):
    """
    Calculate the AUC (Area Under the Curve) for a sequence of predictions and labels, considering valid (non-padded) data.

    Args:
        predictions_all (Tensor): The model's predicted values with shape (batch_size, seq_len, num_skills).
        answers_with_labels (Tensor): Ground truth tensor with shape (batch_size, seq_len, num_skills + 1),
                                     where the last element indicates correctness.
        lengths (Tensor): Lengths of sequences for each batch, to ignore padded values.

    Returns:
        auc (float): AUC score for the batch.
    """
    # Extract problem_ids and labels from the inputs
    predictions, labels = _transform_to_correct_predictions(predictions_all, answers_with_labels)

    # Move tensors to CPU and convert to NumPy arrays for AUC calculation
    predictions = predictions.cpu().detach().numpy()  # Move to CPU first
    labels = labels.cpu().detach().numpy()  # Same for labels
    lengths = lengths.cpu().detach().numpy()

    # Create mask to ignore padded values based on sequence lengths
    mask = np.arange(predictions.shape[1])[None, :] < lengths[:, None]  # Shape: (batch_size, seq_len)

    # Apply the mask to filter valid predictions and labels without using np.nan
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]

    # Calculate AUC if valid data is present
    if valid_predictions.size > 0 and valid_labels.size > 0:
        auc = roc_auc_score(valid_labels, valid_predictions)
    else:
        auc = float('nan')  # Return NaN if no valid data for AUC calculation

    return auc


# Assuming test_loader is your DataLoader for the test set
def evaluate_auc(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    auc_scores = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for skill_sequences, other_sequences, answers, lengths in test_loader:
            # Move inputs and answers to the correct device (GPU/CPU)
            skill_sequences = skill_sequences.to(device)
            other_sequences = other_sequences.to(device)
            answers = answers.to(device)
            lengths = lengths.to('cpu')

            # Forward pass through the model
            predictions = model(skill_sequences, other_sequences, lengths)  # Shape (batch_size, seq_len, num_items)

            # Calculate AUC for the current batch
            auc = calculate_auc(predictions, answers, lengths)
            auc_scores.append(auc)

    # Calculate the average AUC over all batches
    average_auc = sum(auc_scores) / len(auc_scores)
    return average_auc


def _transform_to_correct_predictions(predictions_all, answers_with_labels):
    # Extract problem IDs (skills) and correctness labels from the inputs
    problem_ids = answers_with_labels[..., :-1]  # Shape: (batch_size, seq_len, num_skills)
    labels = answers_with_labels[..., -1]   # Shape: (batch_size, seq_len)

    # Element-wise multiplication of predictions and problem IDs to select relevant predictions
    product = predictions_all * problem_ids  # Shape: (batch_size, seq_len, num_skills)

    # Sum over the skill dimension to aggregate predictions for each sequence step
    result_sum = product.sum(dim=2)  # Shape: (batch_size, seq_len)

    # Count the number of relevant skills (ones) for each step to use as a normalization factor
    num_ones = problem_ids.sum(dim=2)  # Shape: (batch_size, seq_len)

    # Avoid division by zero by adding a small constant to the normalization factor
    normalization_factor = num_ones + 1e-8  # Adding a small constant for numerical stability

    # Normalize the summed result by the number of ones (relevant skills)
    result = result_sum / normalization_factor  # Shape: (batch_size, seq_len)

    return result, labels
