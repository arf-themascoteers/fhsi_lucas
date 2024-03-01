import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVIBandsOnly(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(2,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        band_8 = x[:,7:8]
        band_4 = x[:,3:4]
        x_short = torch.hstack((band_4, band_8))
        return self.linear(x_short)
