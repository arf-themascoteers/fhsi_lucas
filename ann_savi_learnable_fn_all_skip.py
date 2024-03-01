import torch
import torch.nn as nn
from ann_savi_learnable_fn_all import ANNSAVILearnableFnAll


class ANNSAVILearnableFnAllSkip(ANNSAVILearnableFnAll):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.linear = nn.Sequential(
            nn.Linear(15,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        savi_val = self.savi(x)
        x = torch.hstack((x, savi_val, self.fn_all))
        return self.linear(x)





