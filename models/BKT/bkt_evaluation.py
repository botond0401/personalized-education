"""
bkt_evaluation.py

This module provides functionality for evaluating the performance of a 
Bayesian Knowledge Tracing (BKT) model. It includes methods for calculating the
AUC evaluation metric based on the model predictions and user answer sequences.

Functions:
- calculate_auc(sequences: list[list[int]], predictions: list[float]) -> float:
    Calculates the AUC score for the given sequences and predictions.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(sequences: list[list[int]], predictions: list[float]) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the given sequences and predictions.

    Parameters:
    - sequences: A list of sequences of user answers (0 or 1).
    - predictions: A list of predicted next observation values (0 or 1).

    Returns:
    - auc_score: The AUC score (float) for the provided sequences and predictions.
    """
    if isinstance(sequences[0], list):
        sequences = np.concatenate(sequences)

    # Check for unique classes in sequences
    unique_classes = np.unique(sequences)
    if len(unique_classes) < 2:
        return 0.5  # Return a neutral score if no variability in data
    return float(roc_auc_score(sequences, predictions))
