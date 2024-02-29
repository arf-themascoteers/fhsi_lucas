import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler


class DSManager:
    def __init__(self, csv, folds=10):
        self.folds = folds
        torch.manual_seed(0)
        self.df = pd.read_csv(csv)

        for col in self.df.columns:
            scaler = MinMaxScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])

        self.data = self.df.sample(frac=1).to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.df)):
            train_data = self.data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.data[test_index]
            train_x = train_data[:,1:]
            train_y = train_data[:,0]
            test_x = test_data[:,1:]
            test_y = test_data[:,0]
            validation_x = validation_data[:,1:]
            validation_y = validation_data[:,0]

            yield train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_folds(self):
        return self.folds


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from soil_dataset import SoilDataset
    dm = DSManager("data/mangrove.csv")
    for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(dm.get_k_folds()):
        ds = SoilDataset(train_x, train_y)
        dataloader = DataLoader(ds, batch_size=500, shuffle=True)
        for batch_number, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            break
        break


