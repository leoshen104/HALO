from __future__ import annotations

import random

import numpy as np


def set_deterministic_seed(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed
