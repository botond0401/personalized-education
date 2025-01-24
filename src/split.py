import random

def split(keys, val_ratio=0.2):
    """Splits keys into training and validation sets based on a specified validation ratio"""
    random.shuffle(keys)
    split_idx = int(len(keys) * (1 - val_ratio))
    train_idx = keys[:split_idx]
    val_idx = keys[split_idx:]
    return train_idx, val_idx
