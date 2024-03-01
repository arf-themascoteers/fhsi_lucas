import torch
import torch.nn as nn
from ann_savi import ANNSAVI


class ANNSAVILearnableFn(ANNSAVI):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds, torch.tensor(0.5))
        self.linear1 = nn.Sequential(
            nn.Linear(2,20),
            nn.LeakyReLU(),
            nn.Linear(20, 1)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        band_8 = x[:,7:8]
        band_4 = x[:,3:4]
        band_4_8 = torch.hstack((band_4, band_8))
        self.L = torch.mean(self.linear1(band_4_8))
        savi_val = self.savi(x)
        return self.linear2(savi_val)

