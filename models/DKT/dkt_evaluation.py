import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def calculate_loss(predictions, answers):
    """
    Calculate binary cross-entropy loss for a sequence of predictions and labels,
    considering the task index (problem_id) for each student.

    Args:
        predictions (Tensor): Raw output logits from the model, shape (batch_size, seq_len, num_items).
        answers (Tensor): Input tensor of shape (batch_size, seq_len, 2), where each entry is [problem_id, label].

    Returns:
        loss (Tensor): Total binary cross-entropy loss for the batch.
    """
    # Extract problem_ids and labels from the inputs
    problem_ids = answers[..., 0].long()  # Shape: (batch_size, seq_len)
    labels = answers[..., 1]  # Shape: (batch_size, seq_len)

    # Modify problem_ids by subtracting 1 and clamping negative values to 0
    problem_ids = problem_ids - 1
    problem_ids = torch.clamp(problem_ids, min=0)  # Ensure no problem_id is negative


    # Gather the logits for the correct problem_id for each student and time step
    selected_logits = predictions.gather(2, problem_ids.unsqueeze(-1))  # Shape (batch_size, seq_len, 1)

    # Squeeze to remove the last dimension (as we have one probability per student per task)
    selected_logits = selected_logits.squeeze(-1)  # Shape (batch_size, seq_len)

    # Calculate the BCE loss
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    # Compute the loss between selected logits and the corresponding labels
    loss = bce_loss(selected_logits, labels)

    return loss


def calculate_auc(predictions, answers):
    """
    Calculate the AUC (Area Under the Curve) for a sequence of predictions and labels.

    Args:
        predictions (Tensor): A tensor of probabilities, shape (batch_size, seq_len, num_items),
                              where each entry represents the probability of correctly answering
                              a specific problem.
        answers (Tensor): Input tensor of shape (batch_size, seq_len, 2), where each entry is [problem_id, label].
                          - problem_id: The ID of the problem.
                          - label: Binary label (0 or 1), indicating if the answer was correct.

    Returns:
        auc (float): AUC score for the batch.
    """
    # Extract problem_ids and labels from the inputs
    problem_ids = answers[..., 0].long()  # Shape: (batch_size, seq_len)
    labels = answers[..., 1].long()  # Shape: (batch_size, seq_len), assumed to be 0 or 1

    # Modify problem_ids by subtracting 1 and clamping negative values to 0
    problem_ids = problem_ids - 1
    problem_ids = torch.clamp(problem_ids, min=0)  # Ensure no problem_id is negative

    # Gather the logits for the correct problem_id for each student and time step
    selected_predictions = predictions.gather(2, problem_ids.unsqueeze(-1))  # Shape (batch_size, seq_len, 1)

    # Squeeze to remove the last dimension (as we have one probability per student per task)
    selected_predictions = selected_predictions.squeeze(-1)  # Shape (batch_size, seq_len)

    # Flatten the tensors for AUC computation
    labels_flat = labels.view(-1).cpu().numpy()
    probabilities_flat = selected_predictions.view(-1).cpu().detach().numpy()

    # Calculate AUC using sklearn
    auc = roc_auc_score(labels_flat, probabilities_flat)

    return auc


# Assuming test_loader is your DataLoader for the test set
def evaluate_auc(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    auc_scores = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, answers, lengths in test_loader:
            # Move inputs and answers to the correct device (GPU/CPU)
            inputs, answers = inputs.to(device), answers.to(device)
            lengths = lengths.to('cpu')

            # Forward pass through the model
            predictions = model(inputs, lengths)  # Shape (batch_size, seq_len, num_items)

            # Calculate AUC for the current batch
            auc = calculate_auc(predictions, answers)
            auc_scores.append(auc)

    # Calculate the average AUC over all batches
    average_auc = sum(auc_scores) / len(auc_scores)
    return average_auc
