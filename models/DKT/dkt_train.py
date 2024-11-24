from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from .dkt_evaluation import calculate_loss
from .dkt_model import DKT
from .dkt_process import process


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
    model = DKT(**model_params).to(device)

    # Initialize the optimizer with the model's parameters and specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop for the specified number of epochs
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}\n")

        # Training phase: Update the model using the training data
        process(model, train_loader, calculate_loss, device, optimizer)

    # Validation phase: Evaluate the model on the validation dataset (if provided)
    if val_loader is not None:
        _, val_auc = process(model, val_loader, calculate_loss, device)
    else:
        # Use training data for evaluation if no validation DataLoader is provided
        print("No validation loader provided. Using training data for evaluation.")
        _, val_auc = process(model, train_loader, calculate_loss, device)

    # Return the trained model and the best validation AUC achieved
    return model, val_auc
