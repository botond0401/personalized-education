
from torch.utils.data import Dataset


class AnswerSet(Dataset):
    def __init__(self, user_dict):

      self.user_ids = list(user_dict.keys())

      # storing a list of the answers
      user_sequences = user_dict.values()
      self.inputs = [user_sequence[:-1] for user_sequence in user_sequences]

      # storing a list of their labels
      self.targets = [user_sequence[1:] for user_sequence in user_sequences]

    def __getitem__(self, indices):
      return self.inputs[indices], self.targets[indices]

    def __len__(self):
        return len(self.inputs)
    