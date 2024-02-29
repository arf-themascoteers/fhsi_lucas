import torch.nn as nn
from ann_base import ANNBase


class ANNSimple(ANNBase):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(train_x.shape[1],20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        return self.linear(x)

