import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import utils
from soil_dataset import SoilDataset


class DSManager:
    def __init__(self, folds=10, feature_set=None):        
        self.folds = folds
        self.feature_set = feature_set
        
        if self.feature_set is None:
            self.feature_set = utils.get_all_features()
        
        torch.manual_seed(0)
        
        df = pd.read_csv(utils.get_data_file())
        self.derived_columns = []
        for col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])

        df, derived_column, base_columns = DSManager.filter(df, self.feature_set)
        self.data = df.sample(frac=1).to_numpy()

    @staticmethod
    def filter(df, feature_set):
        all_base_columns = utils.get_all_features()
        base_columns = []
        derived_columns = []
        for feature in feature_set:
            if feature in all_base_columns:
                base_columns.append(feature)
            else:
                derived_columns.append(feature)

        for derived_column in derived_columns:
            new_values = DSManager.derive(df, derived_column)
            df.insert(len(df.columns) - 1, derived_column, new_values)

        for a_base_column in all_base_columns:
            if a_base_column not in base_columns:
                df.drop(columns=[a_base_column], axis=1, inplace=True)

        return df, derived_column, base_columns

    @staticmethod
    def derive(df, column_name):
        b4 = df["b4"]
        b8 = df["b8"]
        if column_name == "ndvi":
            den = (b8+b4)
            den[den==0]=0.0001
            new_values = (b8-b4)/den
        return new_values

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.data)):
            train_data = self.data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.data[test_index]
            train_x = train_data[:, 0:-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, 0:-1]
            validation_y = validation_data[:, -1]

            yield SoilDataset(train_x, train_y), \
                SoilDataset(test_x, test_y), \
                SoilDataset(validation_x, validation_y)

    def get_folds(self):
        return self.folds


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from soil_dataset import SoilDataset
    dm = DSManager(3,["ndvi","b1","b4"])
    for fold_number, (dtrain, dtest, dval) in enumerate(dm.get_k_folds()):
        dataloader = DataLoader(dtrain, batch_size=500, shuffle=True)
        for batch_number, (x, y) in enumerate(dataloader):
            print(x.shape)
            print(y.shape)
            break
        break


