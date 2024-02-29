from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from soil_dataset import SoilDataset
from ds_manager import DSManager

dm = DSManager("data/mangrove.csv")
r2s = []

for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(dm.get_k_folds()):
    ds = SoilDataset(train_x, train_y)
    dataloader = DataLoader(ds, batch_size=500, shuffle=True)
    for batch_number, (x, y) in enumerate(dataloader):
        model_instance = LinearRegression()
        model_instance = model_instance.fit(train_x, train_y)
        r2s.append(model_instance.score(test_x, test_y))

print(r2s)
print(sum(r2s)/len(r2s))


