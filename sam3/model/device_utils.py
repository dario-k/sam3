import torch
import logging


def get_optimal_device():
    """
    Returns the best available device in order of priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (Fallback)
    """
    allow_mps = False
    if torch.cuda.is_available():
        return torch.device("cuda")

    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Optional: Check if built with MPS support to avoid edge cases
        if torch.backends.mps.is_built():
            return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_optimal_device()
print(f"Using {DEVICE=}")