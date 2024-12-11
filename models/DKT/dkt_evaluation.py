import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_DKT_loss(predictions_all, answers_with_labels):
    """
    Calculate the binary cross-entropy loss for a sequence of predictions and labels,
    taking into account the problem ID (task index) for each student.

    Args:
        predictions_all (Tensor): The model's predicted values with shape (batch_size, seq_len, num_skills).
        answers_with_labels (Tensor): Ground truth tensor with shape (batch_size, seq_len, num_skills + 1),
                                     where the last dimension contains problem IDs and correctness labels.
                                     The last element of each entry indicates whether the response was correct (1) or not (0).

    Returns:
        loss (float): The total binary cross-entropy loss for the batch.
    """
    # Extract problem_ids and labels from the inputs
    result, labels = _transform_to_correct_predictions(predictions_all, answers_with_labels)
    # Compute binary cross-entropy loss between the normalized result and the correctness labels
    loss = F.binary_cross_entropy(result, labels)

    return loss.item()


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
    predictions = predictions.detach().numpy()
    labels = labels.detach().numpy()
    lengths = lengths.detach().numpy()
    
    # Create mask to ignore padded values based on sequence lengths
    mask = np.arange(predictions.shape[1])[None, :] < lengths[:, None]  # Shape: (batch_size, seq_len)

    # Apply mask to filter valid predictions and labels
    masked_predictions = np.where(mask, predictions, np.nan)  # Set padded positions to np.nan
    masked_labels = np.where(mask, labels, np.nan)  # Set padded positions to np.nan

    # Flatten the arrays to compute AUC only on non-padded data
    flattened_predictions = masked_predictions[~np.isnan(masked_predictions)]
    flattened_labels = masked_labels[~np.isnan(masked_labels)]

    # Calculate AUC if valid data is present
    if flattened_predictions.size > 0 and flattened_labels.size > 0:
        auc = roc_auc_score(flattened_labels, flattened_predictions)
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
