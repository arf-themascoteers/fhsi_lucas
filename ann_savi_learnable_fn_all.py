import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVILearnableFnAll(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear1 = nn.Sequential(
            nn.Linear(13,20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )

    def get_L(self, x=None):
        return torch.mean(self.linear1(x))




