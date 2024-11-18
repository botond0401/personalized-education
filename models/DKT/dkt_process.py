from tqdm import tqdm
import torch
import sys
from .dkt_evaluation import calculate_auc

def process(model, loader, criterion, device, optim=None):
    """
    Process the data in the given loader for either training or evaluation.

    Args:
        model: The model to be used for predictions.
        loader: The DataLoader providing the data.
        criterion: The loss function.
        optim: The optimizer (if training).

    Returns:
    """
    # Set model to training or evaluation mode
    if optim is not None:
        model.train()
        desc = 'Training'
    else:
        model.eval()
        desc = 'Evaluation'

    losses = []
    auc_scores = []

    with torch.no_grad() if optim is None else torch.enable_grad():

      # Iterate through the DataLoader with tqdm for progress tracking
      for batch_idx, (sequences, labels, lengths) in tqdm(enumerate(loader),
                                                          file=sys.stdout,
                                                          unit=' batches',
                                                          desc=desc):
          # Move sequences and labels to the appropriate device
          sequences = sequences.to(device)
          labels = labels.to(device)
          lengths = lengths.to('cpu')

          # Forward pass
          outputs = model(sequences, lengths)
          loss = criterion(outputs, labels)
          auc = calculate_auc(outputs, labels)

          if optim is not None:  # Only during training
              optim.zero_grad()  # Reset gradients
              loss.backward()  # Backpropagation
              optim.step()  # Update parameters
              torch.cuda.empty_cache()

          losses.append(loss.item())  # Accumulate loss
          auc_scores.append(auc.item())  # Accumulate loss

    return sum(losses), sum(auc_scores) / len(auc_scores)
