import torch
import torch.nn as nn
from ann_base import ANNBase
import utils


class ANNSAVI(ANNBase):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.L = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.linear = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        savi_val = self.savi(x)
        return self.linear(savi_val)

    def savi(self, x):
        band_8 = x[:,7]
        band_4 = x[:,3]
        savi = ((band_8-band_4)/(band_8+band_4+self.L))*(1+self.L)
        return savi.reshape(-1,1)

    def verbose_after(self, x, y):
        savi = self.savi(x).reshape(-1)
        pc = utils.calculate_pc(y.detach().cpu().numpy(), savi.detach().cpu().numpy())
        print(f" L: {self.L.item()}, PC: {pc} ", end="")

    def pc(self, ds):
        x = ds.x.to(self.device)
        y = ds.y.numpy()
        savi = self.savi(x).reshape(-1)
        return utils.calculate_pc(y, savi.detach().cpu().numpy())
