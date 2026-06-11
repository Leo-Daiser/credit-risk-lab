"""Random state utilities for reproducibility."""
from app.config import get_config


def get_random_state(seed: int | None = None) -> int:
    """Get random state from config or override."""
    if seed is not None:
        return seed
    config = get_config()
    return config.training.random_state


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    import numpy as np
    np.random.seed(seed)
