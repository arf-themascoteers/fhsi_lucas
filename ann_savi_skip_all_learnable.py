import torch
import torch.nn as nn
from ann_savi_skip_all import ANNSAVISkipAll


class ANNSAVISkipAllLearnable(ANNSAVISkipAll):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.L = nn.Parameter(torch.tensor(0.5))
