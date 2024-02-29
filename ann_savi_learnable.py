import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVILearnable(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.L = nn.Parameter(torch.tensor(0.5))
