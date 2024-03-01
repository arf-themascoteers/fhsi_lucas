import torch
import torch.nn as nn
from ann_savi_skip import ANNSAVISkip


class ANNSAVISkipLearnable(ANNSAVISkip):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds, nn.Parameter(torch.tensor(0.5)))

