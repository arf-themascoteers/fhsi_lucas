import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVILearnableBI(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds, torch.tensor(0.5))
        self.linear_l = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )

    def get_L(self, x=None):
        return torch.mean(self.linear_l(self.get_bi(x)))

    def get_bi(self, x):
        band_3 = x[:, 2]
        band_4 = x[:, 3]
        self.bi = torch.sqrt((torch.square(band_3) + torch.square(band_4)) / 2).reshape(x.shape[0], 1)
        return self.bi



