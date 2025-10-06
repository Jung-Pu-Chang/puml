import numpy as np
import random


class BaseWithSeed:
    """基底 class，統一處理 random seed"""

    def __init__(self, seed: int = 17):
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"[{self.__class__.__name__}] All seeds set to {self.seed}")
