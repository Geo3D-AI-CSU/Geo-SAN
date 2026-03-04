import torch
import random
import numpy as np

def select_device(desired_gpu=2):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        if desired_gpu < num_gpus:
            # Use torch.cuda.set_device to set the default GPU.
            torch.cuda.set_device(desired_gpu)
            # Return the device object
            device = torch.device(f'cuda:{desired_gpu}')
            print(f"Use of equipment: {device}")
        else:
            device = torch.device('cpu')
    else:
        print("CUDA is unavailable. Using CPU.")
        device = torch.device('cpu')

    return device

def set_random_seed(seed=42):
    """
    Set a random seed to ensure the reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If multiple GPUs are used
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
