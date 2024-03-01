import torch
import torch.nn as nn
from ann_base import ANNBase
import utils


class ANNSAVI(ANNBase):
    def __init__(self, train_ds, test_ds, validation_ds, L=None):
        super().__init__(train_ds, test_ds, validation_ds)
        if L is None:
            self.L = torch.tensor(0.5)
        else:
            self.L = L
        self.linear = nn.Sequential(
            nn.Linear(1,20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10,1)
        )

    def get_L(self, x=None):
        return self.L

    def forward(self,x):
        savi_val = self.savi(x)
        return self.linear(savi_val)

    def savi(self, x):
        band_8 = x[:,7]
        band_4 = x[:,3]
        l = self.get_L(x)
        self.savi_val = ((band_8-band_4)/(band_8+band_4+l))*(1+l)
        self.savi_val = self.savi_val.reshape(-1,1)
        return self.savi_val

    def verbose_after(self, ds):
        x = ds.x.to(self.device)
        pc = self.pc(ds)
        print(f" L: {self.get_L(x).item()}, PC: {pc} ", end="")

    def pc(self, ds):
        y = ds.y.numpy()
        return utils.calculate_pc(y, self.savi_val.reshape(-1).detach().cpu().numpy())
