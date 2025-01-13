from typing import Dict, Any
import sys
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from .dkt_train import train_dkt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_loader_helper import SequenceDataset, collate_batch


def k_fold_cv_dkt(
    num_folds,
    model_params: Dict[str, Any],
    lr: float,
    num_epochs: int,
    device: torch.device,
    train_dict: Dict[int, Any],
    batch_size: int = 100,
    num_workers: int = 2
) -> float:
    """
    Perform k-fold cross-validation for the Deep Knowledge Tracing (DKT) model.

    Args:
        num_folds (int): Number of folds for cross-validation.
        model_params (Dict[str, Any]): Parameters for initializing the DKT model.
        lr (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs for training in each fold.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the training on.
        train_dict (Dict[int, Any]): Dictionary mapping keys to data for training.
        batch_size (int, optional): Batch size for training and validation. Defaults to 100.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.

    Returns:
        float: The average validation AUC across all folds.

    Notes:
        - Assumes that the `train_dict` keys can be split into train and validation sets.
        - Requires an `AnswerSet` dataset and the `train_dkt` training function.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    train_keys = list(train_dict.keys())

    val_auc_list = []

    for fold, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_keys)):
        print(f"\nFold {fold+1}/{num_folds}")
        fold_train_keys = [train_keys[i] for i in fold_train_idx]
        fold_val_keys = [train_keys[i] for i in fold_val_idx]

        fold_train_dict = {key: train_dict[key] for key in fold_train_keys}
        fold_train_dict = dict(sorted(fold_train_dict.items(), key=lambda item: len(item[1])))

        fold_val_dict = {key: train_dict[key] for key in fold_val_keys}
        fold_val_dict = dict(sorted(fold_val_dict.items(), key=lambda item: len(item[1])))

        fold_train_dataset = SequenceDataset(fold_train_dict)
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, collate_fn=collate_batch, pin_memory=True, num_workers=num_workers)

        fold_val_dataset = SequenceDataset(fold_val_dict)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, collate_fn=collate_batch, pin_memory=True, num_workers=num_workers)

        _, list_val_loss, list_val_auc = train_dkt(
            model_params, lr, num_epochs, device,
            fold_train_loader, fold_val_loader
            )
        
        val_auc = list_val_auc[-1]

        val_auc_list.append(val_auc)
        print(f'For the {fold+1}. fold AUC is {val_auc}.')

    val_auc_avg = sum(val_auc_list) / num_folds

    return val_auc_avg
