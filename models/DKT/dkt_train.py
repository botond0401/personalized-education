from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from .dkt_model import DKT
from tqdm import tqdm
import sys
from .dkt_evaluation import calculate_auc, calculate_DKT_loss


def process(model, loader, device, optim=None):
    """
    Process the data in the given loader for either training or evaluation.

    Args:
        model: The model to be used for predictions.
        loader: DataLoader providing batches of input data and labels.
        device: The device (CPU or GPU) to run the computation on.
        optim: Optimizer for training (if provided). If None, the function performs evaluation.

    Returns:
        total_loss (float): The sum of all batch losses.
        average_auc (float): The average AUC score across all batches.
    """
    # Set model to training or evaluation mode
    if optim is not None:
        model.train()
        desc = 'Training'
    else:
        model.eval()
        desc = 'Evaluation'

    total_loss = 0
    total_auc = 0
    total_samples = 0


    model = model.to(device)

    with torch.no_grad() if optim is None else torch.enable_grad():

      # Iterate through the DataLoader with tqdm for progress tracking
      for skill_sequences, other_sequences, labels, lengths in tqdm(loader,
                                                                    file=sys.stdout,
                                                                    unit=' batches',
                                                                    desc=desc):
          # Move sequences and labels to the appropriate device
          skill_sequences = skill_sequences.to(device)
          other_sequences = other_sequences.to(device)
          labels = labels.to(device)
          lengths = lengths.to('cpu')

          # Forward pass
          outputs = model(skill_sequences, other_sequences, lengths)
          loss = calculate_DKT_loss(outputs, labels, lengths)
          auc = calculate_auc(outputs, labels, lengths)

          if optim is not None:  # Only during training
              optim.zero_grad()  # Reset gradients
              loss.backward()  # Backpropagation
              optim.step()  # Update parameters
              torch.cuda.empty_cache()

          batch_size = skill_sequences.size(0)  # Get the number of samples in the current batch

          # Weight the loss by the batch size
          total_loss += loss.item() * batch_size  # Accumulate weighted loss
          total_auc += auc * batch_size  # Accumulate weighted AUC
          total_samples += batch_size 

    return total_loss / total_samples, total_auc / total_samples


def train_dkt(
    model_params: Dict[str, Any],
    lr: float,
    num_epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> Tuple[torch.nn.Module, float]:
    """
    Trains a Deep Knowledge Tracing (DKT) model using the provided parameters.

    Args:
        model_params (dict): A dictionary of model parameters to initialize the DKT model.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the training on.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset. If None,
                                           training data is used for evaluation.

    Returns:
        Tuple[torch.nn.Module, float]: The trained DKT model and the best validation AUC achieved.

    Notes:
        - If `val_loader` is not provided, the training data is used for evaluation, which may
          lead to overestimation of the performance.
    """
    # Initialize the model and move it to the specified device
    model = DKT(**model_params)

    # Initialize the optimizer with the model's parameters and specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    list_val_loss = []
    list_val_auc = []

    # Training loop for the specified number of epochs
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}\n")

        # Training phase: Update the model using the training data
        process(model, train_loader, device, optimizer)

        # Validation phase: Evaluate the model on the validation dataset (if provided)
        if val_loader is not None:
            val_loss, val_auc = process(model, val_loader, device)
        else:
            # Use training data for evaluation if no validation DataLoader is provided
            print("No validation loader provided. Using training data for evaluation.")
            val_loss, val_auc = process(model, train_loader, device)
        list_val_loss.append(val_loss)
        list_val_auc.append(val_auc)
        print(f'For the {epoch}. epoch AUC is {val_auc}, loss is {val_loss}.')

    # Return the trained model and the best validation AUC achieved
    return model, list_val_loss, list_val_auc
