import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVISkipAll(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(4,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        savi_val = self.savi(x)
        band_8 = x[:,7:8]
        band_4 = x[:,3:4]
        l = self.L.repeat(x.shape[0], 1)
        x_short = torch.hstack((band_4, band_8, l))
        x = torch.hstack((x_short, savi_val))
        return self.linear(x)
