import torch
import random
import numpy as np

def set_seed(seed):
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed (CPU)
    torch.manual_seed(seed)
    
    # PyTorch random seed (GPU, if applicable)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
