import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, root_mean_squared_error
import utils


class ANNBase(nn.Module):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__()
        self.verbose = True
        self.TEST = False
        self.device = utils.get_device()
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.validation_ds = validation_ds
        self.num_epochs = 2000
        if utils.is_test():
            self.num_epochs = 3
        self.batch_size = 30000
        self.lr = 0.005

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='mean')
        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        total_batch = len(dataloader)
        for epoch in range(self.num_epochs):
            for batch_number, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)

                if self.verbose:
                    r2_test = r2_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
                    y_all, y_hat_all = self.evaluate(self.validation_ds)
                    r2_validation = r2_score(y_all, y_hat_all)
                    print(f'Epoch:{epoch} (of {self.num_epochs}), Batch: {batch_number+1} of {total_batch}, '
                          f'Loss:{loss.item():.6f}, '
                          f'R2_TRAIN: {r2_test:.3f}, R2_Validation: {r2_validation:.3f}', end=""
                          )
                    self.verbose_after(x, y)
                    print("")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def verbose_after(self, x, y):
        pass

    def evaluate(self, ds):
        batch_size = 30000
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        y_all = np.zeros(0)
        y_hat_all = np.zeros(0)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_all = np.concatenate((y_all, y))
            y_hat_all = np.concatenate((y_hat_all, y_hat))

        return y_all, y_hat_all

    def test(self):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(self.test_ds)
        return y_hat_all

    def metrics(self, ds):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(ds)
        r2 = r2_score(y_all, y_hat_all)
        rmse = root_mean_squared_error(y_all, y_hat_all)
        pc = self.pc(ds)
        return r2, rmse, pc

    def run(self):
        self.train_model()
        return self.metrics(self.test_ds)

    def pc(self, ds):
        y_all, y_hat_all = self.evaluate(ds)
        return utils.calculate_pc(y_all, y_hat_all)