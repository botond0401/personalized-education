import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, user_dict):

      # sorting in order to make apdding more efficient
      self.user_ids = sorted(user_dict.keys(), key=lambda x: len(user_dict[x]))
      self.sequences = [user_dict[user_id] for user_id in self.user_ids]

    def __getitem__(self, indices):
      sequence = self.sequences[indices]
      skill_one_hot_vectors = torch.tensor(np.array([item[0] for item in sequence[:-1]]), dtype=torch.float32)
      if len(sequence[0]) == 2:
          # Convert the list comprehension to a numpy array first, then to a tensor
          additional_features = torch.tensor(np.array([item[1] for item in sequence[:-1]]), dtype=torch.float32)
          additional_features = additional_features.unsqueeze(-1)  # Add the extra dimension
      else:
          # Convert the list comprehension to a numpy array first, then to a tensor
          additional_features = torch.tensor(np.array([list(item[1]) + [item[2]] for item in sequence[:-1]]), dtype=torch.float32)

      # Optimize labels creation
      labels = torch.tensor(np.array([list(item[0]) + [item[-1]] for item in sequence[1:]]), dtype=torch.float32)

      return skill_one_hot_vectors, additional_features, labels

    def __len__(self):
        return len(self.sequences)


def collate_batch(batch):
    # Unpacking batches into sequences and labels
    skill_one_hot_vectors, additional_features, labels = zip(*batch)  # This will now work as expected

    # Convert sequences to padded tensors
    skill_one_hot_vectors_padded = pad_sequence(skill_one_hot_vectors, batch_first=True, padding_value=0)  # Pad inputs to max length
    additional_features_padded = pad_sequence(additional_features, batch_first=True, padding_value=0)  # Pad targets to max length
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)  # Pad targets to max length
    lengths = torch.tensor([len(label) for label in labels])

    return skill_one_hot_vectors_padded, additional_features_padded, labels_padded, lengths
