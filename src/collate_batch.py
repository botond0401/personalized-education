import torch
from torch.nn.utils.rnn import pad_sequence


# Define collate function
def collate_batch(batch):
    # Unpacking batches into sequences and labels
    sequences, labels = zip(*batch)  # This will now work as expected

    # Convert sequences to padded tensors
    sequences_tensor = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], 
                                    batch_first=True, padding_value=0)

    # Pad the labels similarly
    # Convert labels to padded tensors
    labels_tensor = pad_sequence([torch.tensor(label, dtype=torch.float) for label in labels], 
                                 batch_first=True, padding_value=0)  # or another padding value if needed

    # Create lengths tensor for sequences (before padding)
    lengths_tensor = torch.tensor([len(seq) for seq in sequences])

    return sequences_tensor, labels_tensor, lengths_tensor
